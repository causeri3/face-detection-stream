from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

tracker = DeepSort(
    # works way better than default
    embedder='clip_RN50x16' ,
    # clip does not run via my silicon arch, dunno if there is a mps setting, set to true on Windows, see if it works
    embedder_gpu = False)

def get_ids(bbs: list[tuple[list, float, str]], frame: np.ndarray) -> list:
    """
    :param bbs: tuple of ( [left,top,w,h], confidence, detection_class)
    :param frame: np.ndarray"""
    tracks = tracker.update_tracks(bbs, frame=frame)
    ids = []
    for track in tracks:
        if not track.is_confirmed():
            ids.append(None)
            continue
        ids.append(track.track_id)
        # ltrb = track.to_ltrb()
    return ids