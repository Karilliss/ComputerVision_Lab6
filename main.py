import argparse, csv, json, os, subprocess, sys, time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from skimage.feature import Cascade
from skimage import data

try: 
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass(frozen=True)
class Cfg:
    video_path: str = "video.mp4"
    work_w: int = 1280     
    norm_w: int = 960      
    norm_h: int = 540

    face_min: int = 36
    face_max: int = 200
    scale_factor: float = 1.1
    min_neighbors: int = 3
   
    link_dist: float = 0.9      
    link_gap: int = 5          
    min_tracklet: int = 3       
    static_eps: float = 6.0

    n_hist_bins: int = 8
    w_color: float = 1.0
    w_space: float = 0.6
    cut_distance: float = 0.9   
    min_person_frames: int = 25 
    min_coherence: float = 8.0  

    max_interp_gap: int = 25
    smooth_win: int = 9
    min_motion_frames: int = 20
    motion_min_width: int = 90


TOTAL_STAGES = 8


def stage(i, title, total=TOTAL_STAGES):
    print(f"\n=== [{i}/{total}] {title} ===", flush=True)


class Bar:
    """Простий прогрес-бар: Bar(total, 'підпис').update(k) / .done()."""
    def __init__(self, total, label="", width=30):
        self.total = max(int(total), 1); self.label = label; self.width = width
        self.t0 = time.time(); self.last = 0.0

    def update(self, k):
        now = time.time()
        if k < self.total and now - self.last < 0.2:
            return
        self.last = now
        frac = min(k / self.total, 1.0)
        fill = int(self.width * frac)
        eta = (now - self.t0) / frac * (1 - frac) if frac > 0 else 0
        sys.stdout.write(f"\r  {self.label} [{'#' * fill}{'.' * (self.width - fill)}] "
                         f"{k}/{self.total} {frac * 100:5.1f}%  ETA {eta:4.0f}s ")
        sys.stdout.flush()

    def done(self):
        self.update(self.total)
        sys.stdout.write(f"\n  готово за {time.time() - self.t0:.1f} с\n"); sys.stdout.flush()


def find_screen(g, dark=35):
    H, W = g.shape
    dm = ((g < dark) * 255).astype(np.uint8)
    dm = cv2.dilate(dm, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))) 
    bright = cv2.bitwise_not(dm)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        if not (0.08 * W * H < a < 0.6 * W * H):
            continue
        fill = a / (w * h); ar = w / h
        if fill > 0.85 and 1.4 < ar < 2.6:
            if best is None or a > best[0]:
                p = 10
                best = (a, (int(x - p), int(y - p), int(w + 2 * p), int(h + 2 * p)))
    return None if best is None else best[1]


def normalize_stream(path, cfg):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Не вдалося відкрити {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    bar = Bar(total, "кадри")
    frames, rois, prev = [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s = cfg.work_w / f.shape[1]
        sm = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r = find_screen(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)) or prev
        if r is None:                                  
            r = (0, 0, sm.shape[1], sm.shape[0])
        prev = r
        x, y, w, h = [int(v / s) for v in r]
        x, y = max(0, x), max(0, y)
        ix, iy = int(w * 0.03), int(h * 0.03)         
        crop = f[y + iy:y + h - iy, x + ix:x + w - ix]
        frames.append(cv2.resize(crop, (cfg.norm_w, cfg.norm_h), interpolation=cv2.INTER_AREA))
        rois.append(r)
        bar.update(len(frames))
    bar.total = max(len(frames), 1); bar.done()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return np.asarray(frames), rois, fps


def detect_faces(frames, cfg):
    cas = Cascade(data.lbp_frontal_face_cascade_filename())
    clahe = cv2.createCLAHE(2.0, (8, 8))
    bar = Bar(len(frames), "кадри")
    out = []
    for k, f in enumerate(frames, 1):
        g = clahe.apply(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        d = cas.detect_multi_scale(g, scale_factor=cfg.scale_factor, step_ratio=1,
                                   min_size=(cfg.face_min, cfg.face_min),
                                   max_size=(cfg.face_max, cfg.face_max),
                                   min_neighbor_number=cfg.min_neighbors,
                                   intersection_score_threshold=0.5)
        out.append([(q['c'], q['r'], q['width'], q['height']) for q in d])
        bar.update(k)
    bar.done()
    return out


def build_tracklets(dets, cfg):
    """Жадібне зв'язування за відстанню центрів (у частках розміру обличчя) з пропуском до link_gap кадрів."""
    tracks = []  
    for t, boxes in enumerate(dets):
        used = set()
        order = sorted(range(len(boxes)), key=lambda i: -boxes[i][2])   
        for i in order:
            x, y, w, h = boxes[i]
            c = np.array([x + w / 2, y + h / 2])
            best, bd = None, 1e9
            for k, tr in enumerate(tracks):
                if k in used or t - tr['frames'][-1] > cfg.link_gap or tr['frames'][-1] == t:
                    continue
                bx, by, bw, bh = tr['boxes'][-1]
                d = np.linalg.norm(c - np.array([bx + bw / 2, by + bh / 2])) / max(w, bw)
                d *= 1 + 0.15 * (t - tr['frames'][-1] - 1)
                if d < bd:
                    best, bd = k, d
            if best is not None and bd < cfg.link_dist:
                tracks[best]['frames'].append(t); tracks[best]['boxes'].append(boxes[i]); used.add(best)
            else:
                tracks.append({'frames': [t], 'boxes': [boxes[i]]}); used.add(len(tracks) - 1)
    return [tr for tr in tracks if len(tr['frames']) >= cfg.min_tracklet]


def gray_blur(frames):
    return [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32) for f in frames]


def residual(G, i, j, win):
    (dx, dy), _ = cv2.phaseCorrelate(G[i], G[j], win)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    al = cv2.warpAffine(G[j], M, (G[j].shape[1], G[j].shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return np.abs(al - G[i])


def motion_maps(frames):
    G = gray_blur(frames)
    n, (h, w) = len(G), G[0].shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    bar = Bar(n, "кадри")
    maps = []
    for t in range(n):
        acc = np.zeros((h, w), np.float32)
        for a, b in ((t - 2, t), (t, t + 2), (t - 1, t + 1)):
            a = min(max(a, 0), n - 1); b = min(max(b, 0), n - 1)
            if a != b:
                acc += residual(G, a, b, win)
        acc = cv2.GaussianBlur(acc, (0, 0), 6)
        m = int(0.04 * w)                      
        acc[:, :m] = 0; acc[:, -m:] = 0; acc[:int(0.08 * h)] = 0    
        maps.append(acc)
        bar.update(t + 1)
    bar.done()
    return maps

BIG = 5.0


def torso_hist(frame, box, bins):
    x, y, w, h = box
    H, W = frame.shape[:2]
    x0 = int(max(0, x - 0.6 * w)); x1 = int(min(W, x + 1.6 * w))
    y0 = int(min(H - 1, y + 1.1 * h)); y1 = int(min(H, y + 3.0 * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    lab = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [bins] * 3, [0, 256] * 3).ravel()
    s = hist.sum()
    return hist / s if s > 0 else None


def tracklet_features(tr, frames, cfg):
    hs = [h for f, b in zip(tr['frames'], tr['boxes'])
          if (h := torso_hist(frames[f], b, cfg.n_hist_bins)) is not None]
    hist = np.mean(hs, axis=0) if hs else np.ones(cfg.n_hist_bins ** 3) / cfg.n_hist_bins ** 3
    b = np.array(tr['boxes'], float)
    pos = (b[:, :2] + b[:, 2:] / 2).mean(0)
    return hist, pos


def bhattacharyya(p, q):
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q)))))


def cluster_tracklets(tracks, frames, cfg):
    n = len(tracks)
    if n < 2:  
        return None, np.ones(n, dtype=int), np.zeros((n, n))
    feats = [tracklet_features(t, frames, cfg) for t in tracks]
    L = float(np.hypot(frames.shape[2], frames.shape[1]))
    D = np.zeros((n, n))
    fsets = [set(t['frames']) for t in tracks]
    for i in range(n):
        for j in range(i + 1, n):
            d = cfg.w_color * bhattacharyya(feats[i][0], feats[j][0]) \
                + cfg.w_space * np.linalg.norm(feats[i][1] - feats[j][1]) / L * 4
            if len(fsets[i] & fsets[j]) > 2:   
                d = BIG
            D[i, j] = D[j, i] = d
    Z = linkage(squareform(D, checks=False), method='average')
    labels = fcluster(Z, t=cfg.cut_distance, criterion='distance')
    return Z, labels, D

def select_face_persons(tracks, labels, cfg):
    """Кластери з достатньою підтримкою та когерентністю -> особи, підтверджені обличчям."""
    persons = []
    for l in np.unique(labels):
        ts = [t for t, k in zip(tracks, labels) if k == l]
        frs = sorted({f for t in ts for f in t['frames']})
        coh = np.mean([len(t['frames']) for t in ts])
        persons.append(dict(label=int(l), frames=len(frs), coherence=float(coh), n_tracklets=len(ts),
                            boxes={f: b for t in ts for f, b in zip(t['frames'], t['boxes'])}))
    keep = [p for p in persons if p['frames'] >= cfg.min_person_frames and p['coherence'] >= cfg.min_coherence]
    return sorted(keep, key=lambda p: -p['frames']), persons


def face_to_body(b, W, H):
    x, y, w, h = b
    bx, by = x - 0.9 * w, y - 0.25 * h
    bw, bh = 2.8 * w, 5.0 * h
    x0, y0 = max(0, bx), max(0, by); x1, y1 = min(W, bx + bw), min(H, by + bh)
    return np.array([x0, y0, x1 - x0, y1 - y0], float)


def interpolate_track(obs, n, max_gap, win):
    """obs: {frame: array(4)} -> масив (n,4) із NaN поза межами; лінійна інтерполяція прогалин <= max_gap, згладжування."""
    out = np.full((n, 4), np.nan)
    fs = sorted(obs)
    for f in fs:
        out[f] = obs[f]
    for a, b in zip(fs[:-1], fs[1:]):
        if 1 < b - a <= max_gap:
            for k in range(a + 1, b):
                out[k] = out[a] + (out[b] - out[a]) * (k - a) / (b - a)
    for f, step in ((fs[0], -1), (fs[-1], 1)):    
        for k in range(1, 9):
            j = f + step * k
            if 0 <= j < n:
                out[j] = out[f]
    sm = out.copy()
    r = win // 2
    for i in range(n):
        if np.isnan(out[i, 0]):
            continue
        seg = out[max(0, i - r):i + r + 1]
        seg = seg[~np.isnan(seg[:, 0])]
        sm[i] = seg.mean(0)
    return sm


def motion_persons(mmaps, known, W, H, cfg):
    """Особи без розпізнаного обличчя: сегменти колонок із залишковою енергією руху поза вже відомими тілами."""
    n = len(mmaps)
    cand = {}
    for t in range(n):
        m = mmaps[t].astype(np.float32)
        col = m[int(0.15 * H):, :].sum(0)
        col = np.convolve(col, np.ones(25) / 25, mode='same')
        mask = np.zeros(W, bool)
        for tr in known:
            b = tr[t]
            if not np.isnan(b[0]):
                mask[int(max(0, b[0] - 0.15 * b[2])):int(min(W, b[0] + 1.15 * b[2]))] = True
        thr = 0.30 * max(col.max(), 1e-6)
        ok = (col > thr) & ~mask
        segs, s = [], None
        for x in range(W):
            if ok[x] and s is None:
                s = x
            if (not ok[x] or x == W - 1) and s is not None:
                segs.append((s, x)); s = None
        segs = [(a, b) for a, b in segs if b - a >= cfg.motion_min_width]
        if not segs:
            continue
        a, b = max(segs, key=lambda q: col[q[0]:q[1]].sum())

        cw = np.cumsum(col[a:b]); cw = cw / cw[-1]
        a2 = a + int(np.searchsorted(cw, 0.08)); b2 = a + int(np.searchsorted(cw, 0.92))
        a, b = a2, max(b2, a2 + 40)
        row = m[:, a:b].sum(1); row = np.convolve(row, np.ones(15) / 15, mode='same')
        rw = np.cumsum(row); rw = rw / rw[-1]
        y0 = int(np.searchsorted(rw, 0.04)); y1 = int(np.searchsorted(rw, 0.985))
        if y1 - y0 < 40:
            continue
        pad = int(0.08 * (b - a))
        a, b = max(0, a - pad), min(W, b + pad)
        cand[t] = np.array([a, y0, b - a, y1 - y0], float)
    return cand


def filter_motion_track(cand, cfg):
    """Лишаємо лише траєкторію, що тримається близько за x принаймні min_motion_frames кадрів."""
    if len(cand) < cfg.min_motion_frames:
        return {}
    fs = sorted(cand)
    cx = np.array([cand[f][0] + cand[f][2] / 2 for f in fs])
    med = np.median(cx)
    keep = {f: cand[f] for f, c in zip(fs, cx) if abs(c - med) < 160}
    return keep if len(keep) >= cfg.min_motion_frames else {}


COLORS_BGR = [(0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def draw(frame, tracks, sources, f, face_boxes=None):
    im = frame.copy()
    for i, t in enumerate(tracks):
        b = t[f]
        if np.isnan(b[0]):
            continue
        x, y, w, h = map(int, b); c = COLORS_BGR[i % 4]
        cv2.rectangle(im, (x, y), (x + w, y + h), c, 3)
        label = f"P{i + 1} [{sources[i]}]"
        (tw, th), _ = cv2.getTextSize(label, 0, 0.8, 2)
        cv2.rectangle(im, (x, y - th - 10), (x + tw + 8, y), c, -1)
        cv2.putText(im, label, (x + 4, y - 6), 0, 0.8, (0, 0, 0), 2)
    if face_boxes:
        for (x, y, w, h) in face_boxes.get(f, []):
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 1)
    n = sum(1 for t in tracks if not np.isnan(t[f, 0]))
    cv2.putText(im, f"frame {f}  persons: {n}", (10, im.shape[0] - 12), 0, 0.8, (255, 255, 255), 2)
    return im


def write_video(path, frames, tracks, sources, fps, face_boxes=None):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bar = Bar(len(frames), "кадри")
    for f in range(len(frames)):
        vw.write(draw(frames[f], tracks, sources, f, face_boxes))
        bar.update(f + 1)
    bar.done()
    vw.release()


def save_grid(frames, tracks, sources, idx, path, cols=4, face_boxes=None):
    tiles = [cv2.resize(draw(frames[f], tracks, sources, f, face_boxes), (480, 270)) for f in idx]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))


def save_annotated_video(raw, final, frames, tracks, sources, fps, face_boxes, out_dir):
    """Записує анотоване відео; перекодовує в H.264, якщо доступний ffmpeg."""
    write_video(raw, frames, tracks, sources, fps, face_boxes)
    if not os.path.exists(raw) or os.path.getsize(raw) == 0:
        raise IOError("Не вдалося записати відео (VideoWriter). Перевірте права на запис у папку " + out_dir)
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", final], check=True)
        os.remove(raw)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("УВАГА: ffmpeg недоступний - відео збережено без перекодування (MP4V).")
        if os.path.exists(final):
            os.remove(final)
        os.replace(raw, final)
    print("Анотоване відео:", os.path.abspath(final))


def verify(tracks, sources, rois, fps):
    n = len(tracks[0]); rep = {}
    rois = np.array([r for r in rois], float)
    rep['screen_roi_found_all_frames'] = bool(len(rois) == n and not np.isnan(rois).any())
    rep['screen_roi_width_std_px'] = float(rois[:, 2].std())
    per = []
    for i, tr in enumerate(tracks):
        pres = ~np.isnan(tr[:, 0]); cen = tr[:, :2] + tr[:, 2:] / 2
        jumps = np.linalg.norm(np.diff(cen[pres], axis=0), axis=1)
        per.append(dict(id=f"P{i+1}", source=sources[i], coverage_frames=int(pres.sum()),
                        coverage_pct=round(100 * pres.mean(), 1),
                        max_center_jump_px=round(float(jumps.max()), 1) if len(jumps) else 0.0,
                        median_center_jump_px=round(float(np.median(jumps)), 2) if len(jumps) else 0.0))
    rep['per_person'] = per
    cnt = np.sum([~np.isnan(t[:, 0]) for t in tracks], axis=0)
    rep['persons_per_frame_mean'] = round(float(cnt.mean()), 2)
    rep['frames_with_3_persons_pct'] = round(100 * float((cnt == 3).mean()), 1)
    rep['frames_with_0_persons'] = int((cnt == 0).sum())

    def iou(a, b):  
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        inter = ix * iy; u = a[2] * a[3] + b[2] * b[3] - inter
        return inter / u if u > 0 else 0
    mx = 0.0
    for f in range(n):
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                if not np.isnan(tracks[i][f, 0]) and not np.isnan(tracks[j][f, 0]):
                    mx = max(mx, iou(tracks[i][f], tracks[j][f]))
    rep['max_pairwise_iou'] = round(mx, 3)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="video.mp4"); ap.add_argument("--out", default="result")
    ap.add_argument("--use-cache", action="store_true", help="використати збережені рівні 0-1 та карти руху")
    a = ap.parse_args(); cfg = Cfg(video_path=a.video); os.makedirs(a.out, exist_ok=True)
    P = lambda n: os.path.join(a.out, n)
    t0 = time.time()

    if a.use_cache and os.path.exists(P("norm_frames.npy")):
        print("Використовую кеш: рівні 0, 1 та 2б пропущено (завантажено з папки " + a.out + ")")
        frames = np.load(P("norm_frames.npy")); meta = json.load(open(P("meta.json")))
        rois, fps = meta["rois"], meta["fps"]; dets = json.load(open(P("raw_detections.json")))
        mmaps = np.load(P("motion_maps.npy"))
    else:
        stage(1, "Рівень 0: пошук екрана телефону та нормалізація кадрів")
        frames, rois, fps = normalize_stream(cfg.video_path, cfg)
        np.save(P("norm_frames.npy"), frames); json.dump(dict(rois=rois, fps=fps), open(P("meta.json"), "w"))
        stage(2, "Рівень 1: каскадний класифікатор облич (LBP)")
        dets = detect_faces(frames, cfg); json.dump(dets, open(P("raw_detections.json"), "w"))
        stage(3, "Рівень 2б: карти руху (компенсація зсуву камери)")
        mmaps = np.array(motion_maps(frames), dtype=np.float16); np.save(P("motion_maps.npy"), mmaps)
    n, H, W = len(frames), frames.shape[1], frames.shape[2]

    stage(4, "Рівень 2: побудова тубіл (часове зв'язування детекцій)")
    tr = build_tracklets(dets, cfg)
    print(f"  кадрів: {n}, сирих кандидатів: {sum(len(d) for d in dets)}, тубіл: {len(tr)}")

    stage(5, "Рівень 3: ієрархічна кластеризація тубіл")
    Z, lab, D = cluster_tracklets(tr, frames, cfg)
    print(f"  кластерів: {len(set(lab))}")

    stage(6, "Рівень 4: злиття, інтерполяція, згладжування -> ID осіб")
    keep, allp = select_face_persons(tr, lab, cfg)
    tracks, sources = [], []
    for p in keep:
        obs = {f: face_to_body(b, W, H) for f, b in p["boxes"].items()}
        tracks.append(interpolate_track(obs, n, cfg.max_interp_gap, cfg.smooth_win)); sources.append("face")
    cand = motion_persons(mmaps, tracks, W, H, cfg); mt = filter_motion_track(cand, cfg)
    if mt:
        tracks.append(interpolate_track(mt, n, cfg.max_interp_gap, cfg.smooth_win)); sources.append("motion")
    if not tracks:  
        print("УВАГА: осіб не знайдено, у відео будуть лише сирі детекції облич.")
        tracks.append(np.full((n, 4), np.nan)); sources.append("none")
    print(f"  осіб знайдено: {sum(1 for x in sources if x != 'none')} "
          f"(за обличчям: {sources.count('face')}, за рухом: {sources.count('motion')})")

    face_boxes = {f: [tuple(int(v) for v in b) for b in d] for f, d in enumerate(dets)}  

    stage(7, "Запис анотованого відео")
    save_annotated_video(P("annotated_raw.mp4"), P("annotated_people.mp4"),
                         frames, tracks, sources, fps, face_boxes, a.out)

    stage(8, "Збереження результатів: CSV, верифікація, графіки")
    with open(P("detections_persons.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "time_s", "person_id", "source", "x", "y", "w", "h"])
        for f in range(n):
            for i, t in enumerate(tracks):
                if not np.isnan(t[f, 0]):
                    w.writerow([f, round(f / fps, 2), f"P{i+1}", sources[i], *[round(float(v), 1) for v in t[f]]])
    clusters = [dict(label=p["label"], frames=p["frames"], tracklets=p["n_tracklets"],
                     coherence=round(p["coherence"], 1), accepted=p in keep) for p in allp]
    rep = verify(tracks, sources, rois, fps); rep["clusters"] = clusters
    rep["raw_face_candidates"] = int(sum(len(d) for d in dets)); rep["tracklets"] = len(tr)
    rep["runtime_s_this_run"] = round(time.time() - t0, 1)
    json.dump(rep, open(P("verification.json"), "w"), ensure_ascii=False, indent=2)
    save_grid(frames, tracks, sources, list(range(10, n, 33))[:12], P("sample_frames.png"), face_boxes=face_boxes)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n=== ГОТОВО за {time.time() - t0:.1f} с. Результати в папці: {os.path.abspath(a.out)} ===")


if __name__ == "__main__":
    main()