import sys
import time
import copy
import json
import queue
import signal
import getopt
import logging
import threading

import cv2
import imutils
import cherrypy
import numpy as np
from mxnet import nd
from mxnet import gluon
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score


LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

MODES = {
    "adjust",
    "server",
}

LETTERS = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'a',
    11: 'b',
    12: 'c',
    13: 'd',
    14: 'e',
    15: 'f',
}

CONFIG = {
    "camera_url": "",
    "vd_frame_queue_size": 32,
    "vd_frame_rate": 12.,
    "vd_read_frame_retry_times": 5,
    "vd_trim_left": 0.,
    "vd_trim_right": 0.,
    "vd_trim_top": 0.,
    "vd_trim_bottom": 0.,
    "fdscv_padding_left": 10,
    "fdscv_padding_right": 10,
    "fdscv_padding_top": 10,
    "fdscv_padding_bottom": 10,
    "fdscv_min_exposure": 235,
    "fdscv_morph_rect_ksize_width": 20,
    "fdscv_morph_rect_ksize_height": 5,
    "fdscv_dilate_interations": 2,
    "cnt_min_exposure": 235,
    "cnt_morph_rect_ksize_width": 2,
    "cnt_morph_rect_ksize_height": 2,
    "cnt_dilate_interations": 2,
    "cnt_min_width": 1,
    "cnt_max_width": 99,
    "cnt_min_height": 1,
    "cnt_max_height": 99,
    "fifr_min_exposure": 170,
    "fifr_resize_width": 22,
    "fifr_resize_height": 22,
    "lenet16_model_params_path": "models/lenet16/lenet16-0000.params",
    "lenet16_model_symbol_path": "models/lenet16/lenet16-symbol.json",
    "ft_dist_size": 350,
    "ft_predict_threshold": 15,
    "ft_train_interval": 150,
    "ft_max_clusters": 5,
    "ft_dist_factor": 0.1,
    "ml_process_delay": 0.1,
    "log_level": "info",
    "mode": "server",
    "http_server_host": "0.0.0.0",
    "http_server_port": 8080,
}

PROFILES = {
    key: {"type": type(val)} for (key, val) in CONFIG.items()
}

RESULT = {
    "tag": 0,
}

running = True
mainloop_running = True


def find_digit_screen_with_cv(
        orig,
        min_exposure=240,
        morph_rect_ksize_width=180, morph_rect_ksize_height=80,
        dilate_interations=1,
        padding_left=20, padding_right=20, padding_top=20, padding_bottom=20,
):
    OH, OW = orig.shape[:2]
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, min_exposure, 0, cv2.THRESH_TOZERO)
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_rect_ksize_width, morph_rect_ksize_height))
    dlt = cv2.dilate(thresh1, krn, iterations=dilate_interations)

    cnts = imutils.grab_contours(cv2.findContours(dlt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    boxes = [cv2.boundingRect(cnt) for cnt in cnts]
    rects = []
    for (x, y, w, h) in boxes:
        rects.append((
            max(x-padding_left, 0),
            max(y-padding_top, 0),
            min(x+w+padding_right, OW),
            min(y+h+padding_bottom, OH),
        ))
    return rects


def find_contours(
        img,
        min_exposure=230,
        morph_rect_ksize_width=2,
        morph_rect_ksize_height=2,
        dilate_interations=2,
        min_width=10, max_width=999,
        min_height=50, max_height=999,
    ):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, min_exposure, 255, cv2.THRESH_TOZERO)
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_rect_ksize_width, morph_rect_ksize_height))
    dlt = cv2.dilate(thresh1, krn, iterations=dilate_interations)
    contours = imutils.grab_contours(cv2.findContours(dlt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

    digits = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if min_width <= w <= max_width and min_height <= h <= max_height:
            digits.append((x, y, w, h))

    return digits


def load_lenet16(symbol, params):
    return gluon.SymbolBlock.imports(
        symbol,
        ['data'],
        params)


def format_image_for_recognization(
        img,
        min_exposure=220,
        resize_width=22, resize_height=22,
    ):
    global format_image_for_recognization_cnt
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, min_exposure, 0, cv2.THRESH_TOZERO)
    resized = imutils.resize(thresh, height=resize_height)
    if resized.shape[1] > 28:
        resized = imutils.resize(resized, width=resize_width)
    dst = np.zeros(shape=(28, 28))
    src = resized
    startY = int(dst.shape[0]/2-src.shape[0]/2)
    startX = int(dst.shape[1]/2-src.shape[1]/2)
    dst[startY:startY+src.shape[0], startX:startX+src.shape[1]] = src
    return nd.array(dst.reshape(1, 1, 28, 28))


def format_text_factory(dist_size=3000, predict_threshold=300, train_interval=1000, max_clusters=5):
    G = {
        'dist': np.zeros(dist_size),
        'dist_idx': 0,
        'dist_cnt': 0,
        'kmm': None,
        'min_dist_label': None,

    }

    def train_model():
        kmms = []
        X = G['dist'][G['dist'] > 0].reshape(-1, 1)
        for i in range(1, max_clusters+1):
            km = KMeans(n_clusters=i)
            yhat = km.fit_predict(X)
            try:
                score = calinski_harabasz_score(X, yhat)
            except ValueError:
                score = np.inf
            label_means = [(label, X[yhat == label].mean()) for label in set(yhat)]
            min_mean_label, min_mean = min(label_means, key=lambda x: x[1])
            max_mean_label, max_mean = max(label_means, key=lambda x: x[1])
            kmms.append((i, score, min_mean_label, min_mean, max_mean_label, max_mean, km))
        k, scr, G['min_dist_label'], _, _, max_mean, G['kmm'] = min(kmms, key=lambda x: x[1])
        k1mean = kmms[0][3]
        dist_factor = abs(k1mean-max_mean)/max_mean
        if dist_factor < CONFIG["ft_dist_factor"]:
            k, scr, G['min_dist_label'], _, _, _, G['kmm'] = kmms[0]
        logging.debug('k=%s, score=%s, min_dist_label=%s, dist_factor=%s', k, scr, G['min_dist_label'], dist_factor)

    def format_text(ds):
        if len(ds) == 0:
            return ''

        if len(ds) == 1:
            return str(ds[0][4])

        ds = sorted(ds, key=lambda x: x[0])
        x1, y1, w1, h1, dig = ds[0]
        s = str(dig)
        for i in range(1, len(ds)):
            x2, y2, w2, h2, dig = ds[i]
            d = ((((x1+w1)-(x2+w2))**2)+((y1+h1)-(y2+h2))**2)**0.5
            x1, y1, w1, h1 = x2, y2, w2, h2
            G['dist'][G['dist_idx']] = d
            G['dist_idx'] = G['dist_idx'] + 1 if G['dist_idx'] + 1 < dist_size else 0
            G['dist_cnt'] += 1
            if G['dist_cnt'] > predict_threshold:
                if G['kmm'] is None:
                    train_model()
                elif G['dist_cnt'] % train_interval == 0:
                    train_model()

            if G['kmm'] is not None:
                p = G['kmm'].predict([[d]])[0]
                if G['min_dist_label'] != p:
                    s += ' '
                s += str(dig)
            else:
                s = ''
        return s.strip()
    return format_text


def elapsed(t0, name):
    e = (time.time() - t0) * 1000
    logging.debug("elapsed(%s)=%.2fms", name, e)


def read_frame_loop(cap, ch):
    global running

    frm_num = 0
    sample_at = 0
    sample_interval = lambda: 1. / CONFIG["vd_frame_rate"]
    read_frame_retry_times = 0
    while running or mainloop_running:
        ret, frm = cap.read()
        if not ret:
            read_frame_retry_times += 1
            logging.warning("failed to read frame from video capture")
            if read_frame_retry_times > CONFIG["vd_read_frame_retry_times"]:
                logging.error("failed to read frame from video capture too many times")
                stop()
            continue
        read_frame_retry_times = 0
        frm_num += 1
        now = time.time()
        if now - sample_at > sample_interval():
            frm_num += 1
            sample_at = now
            try:
                frm = imutils.resize(frm, height=320)
                ch.put_nowait((frm_num, now, frm))
                # logging.debug("put_frame=%s, shape=(%d,%d)", frm_num, *frm.shape[:2])
            except queue.Full:
                # logging.debug("drops=%d", frm_num)
                pass
    else:
        cleanup(cap)
        logging.debug("read frame loop exit")


class DigitFinderWebService(object):
    @cherrypy.expose
    def config(self, format=""):
        if format == "cmd":
            cfg = copy.deepcopy(CONFIG)
            cfg["mode"] = "server"
            body = " ".join(["--%s='%s'" % (k, v) for (k, v) in cfg.items()])
        elif format == "json":
            body = json.dumps(CONFIG, indent=2, sort_keys=True)
        else:
            body = "<pre>%s</pre>" % ("\n".join(["%s: %s" % (k, v) for (k, v) in CONFIG.items()]))
        return body

    @cherrypy.expose
    def set_config(self, **kwargs):
        for key, val in kwargs.items():
            set_config(key, val)
        return "OK"

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def result(self):
        cherrypy.response.headers["X-Timestamp"] = RESULT.get("timestamp", 0)
        cherrypy.response.headers["X-Tag"] = RESULT.get("tag", 0)
        return {"text": RESULT.get("text", "")}


def mainloop(ch, lenet):
    global mainloop_running

    format_text = format_text_factory(
        dist_size=CONFIG["ft_dist_size"],
        predict_threshold=CONFIG["ft_predict_threshold"],
        train_interval=CONFIG["ft_train_interval"],
        max_clusters=CONFIG["ft_max_clusters"],
    )
    curr = ""

    while running:
        num, ts, frm = ch.get()
        h, w = frm.shape[:2]
        frm = frm[int(h*CONFIG["vd_trim_top"]):int(h*(1-CONFIG["vd_trim_bottom"])),
                  int(w*CONFIG["vd_trim_left"]):int(w*(1-CONFIG["vd_trim_right"]))]

        start = time.time()
        if start - ts > CONFIG["ml_process_delay"]:
            logging.debug("frame out of date")
            continue

        boxes = find_digit_screen_with_cv(
            frm,
            padding_left=CONFIG["fdscv_padding_left"],
            padding_right=CONFIG["fdscv_padding_right"],
            padding_top=CONFIG["fdscv_padding_top"],
            padding_bottom=CONFIG["fdscv_padding_bottom"],
            min_exposure=CONFIG["fdscv_min_exposure"],
            morph_rect_ksize_width=CONFIG["fdscv_morph_rect_ksize_width"],
            morph_rect_ksize_height=CONFIG["fdscv_morph_rect_ksize_height"],
            dilate_interations=CONFIG["fdscv_dilate_interations"],
        )

        for (bStartX, bStartY, bEndX, bEndY) in boxes:
            scr_img = frm[bStartY:bEndY, bStartX:bEndX]
            contours = find_contours(
                scr_img,
                min_exposure=CONFIG["cnt_min_exposure"],
                morph_rect_ksize_width=CONFIG["cnt_morph_rect_ksize_width"],
                morph_rect_ksize_height=CONFIG["cnt_morph_rect_ksize_height"],
                dilate_interations=CONFIG["cnt_dilate_interations"],
                min_width=CONFIG["cnt_min_width"],
                max_width=CONFIG["cnt_max_width"],
                min_height=CONFIG["cnt_min_height"],
                max_height=CONFIG["cnt_max_height"],
            )

            if len(contours) == 0:
                continue

            ds = []
            for (cntX, cntY, cntW, cntH) in contours:
                dig_img = scr_img[cntY:cntY+cntH, cntX:cntX+cntW]
                X = format_image_for_recognization(
                    dig_img,
                    min_exposure=CONFIG["fifr_min_exposure"],
                    resize_width=CONFIG["fifr_resize_width"],
                    resize_height=CONFIG["fifr_resize_height"],
                )
                y = lenet(X).argmax(axis=1)
                dig = LETTERS[int(y.asnumpy()[0])]
                dig = dig if dig != 'f' else '1'
                ds.append([cntX, cntY, cntW, cntH, dig])

            text = format_text(ds)

            if text != "" and curr != text:
                curr = text
                RESULT["text"] = text
                RESULT["timestamp"] = start
                RESULT["tag"] += 1
                logging.debug("num=%s, text=%s, %s", num, text, ", ".join(["%s=(%s,%s)" % (d, w, h) for (x, y, w, h, d) in ds]))
    else:
        mainloop_running = False
        logging.debug("mainloop exit")


def signal_handler(sig, frm):
    stop()


def stop():
    global running

    running = False
    cherrypy.engine.stop()


def cleanup(cap):
    cap.release()


def http_server_loop():
    cherrypy.config.update({
        "environment": "embedded",
        "server.socket_host": CONFIG["http_server_host"],
        "server.socket_port": CONFIG["http_server_port"],

    })
    cherrypy.tree.mount(DigitFinderWebService())
    cherrypy.engine.start()


def start():
    global running

    cap = cv2.VideoCapture(CONFIG["camera_url"])
    if not cap.isOpened():
        raise RuntimeError("failed to open camera url")
    logging.debug("open video capture")

    lenet16 = load_lenet16(CONFIG["lenet16_model_symbol_path"], CONFIG["lenet16_model_params_path"])
    logging.debug("load lenet16 model")

    frm_ch = queue.Queue(CONFIG["vd_frame_queue_size"])

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    threads = []

    logging.debug("starting main loop")
    th = threading.Thread(target=mainloop, kwargs={"ch": frm_ch, "lenet": lenet16})
    th.start()
    threads.append(th)

    logging.debug("starting read frame loop")
    th = threading.Thread(target=read_frame_loop, kwargs={"cap": cap, "ch": frm_ch})
    th.start()
    threads.append(th)

    logging.debug("starting http server")
    threading.Thread(target=http_server_loop).start()

    # logging.debug("starting main loop")
    # mainloop(frm_ch, lenet16)

    # logging.debug("starting read frame loop")
    # read_frame_loop(cap=cap, ch=frm_ch)

    for th in threads:
        th.join()


def set_config(key, val):
    if key not in CONFIG:
        raise RuntimeError("unexpected config key")
    CONFIG[key] = PROFILES[key].get("type", str)(val)


def load_config(config_file):
    with open(config_file) as f:
        for k, v in json.loads(f.read()).items():
            CONFIG[k] = v


def usage():
    print("usage: ")
    sys.exit(1)


def main(argv):
    args, opts = getopt.getopt(
        argv,
        "hc:",
        ["help", "config="] + [k+"=" for k in CONFIG.keys()])

    for (key, val) in args:
        if key.strip('-') in CONFIG:
            set_config(key.strip('-'), val)
        elif key in ['-h', '--help']:
            usage()
        elif key in ['-c', '--config']:
            load_config(val)

    logging.basicConfig(level=LOG_LEVEL[CONFIG["log_level"]])
    if CONFIG["mode"] == "adjust":
        CONFIG["cnt_min_width"] = 1
        CONFIG["cnt_max_width"] = 9999
        CONFIG["cnt_min_height"] = 1
        CONFIG["cnt_max_height"] = 9999

    logging.debug("config: %s", CONFIG)

    start()


if __name__ == '__main__':
    main(sys.argv[1:])
