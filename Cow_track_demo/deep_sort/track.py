# vim: expandtab:ts=4:sw=4
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    Tentative = 1
    Confirmed = 2
    # There are 3 sub-status in confirmed status. They are Detecting, Mismatched, Matched
    Deleted = 3


class ConfirmedState:
    Detecting = 1
    Re_Identify = 2
    Matched = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.cow_id = 'Detecting'
        self.predicts = []
        self.confidences = []
        # self.cow_id = cow_id
        # self.reg_conf = reg_conf

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self._confirmed_state = None
        # self.tracked_frame=1
        self.label = None

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0

        # if detection.reg_conf>max(0.4, self.reg_conf):
        #     self.cow_id = detection.cow_id
        #     self.reg_conf = detection.reg_conf
        # if self.reg_conf<0.4:
        #     self.cow_id = str(f'UK#{self.track_id}')
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        if self.state == TrackState.Confirmed:
            # 对Unknown的牛， 不进行 re-id
            if 'Unknown' in self.cow_id: return
            if self.cow_id == 'Detecting':
                self._confirmed_state = ConfirmedState.Detecting
            else:
                # self._confirmed_state = ConfirmedState.Re_Identify if self.hits%100==0 else ConfirmedState.Matched
                if self.hits%100==0:
                    self._confirmed_state = ConfirmedState.Re_Identify
                    self.cow_id+='Re-ID'

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def is_detecting(self):
        return self.state == TrackState.Confirmed and self._confirmed_state == ConfirmedState.Detecting
        # return 'Detecting' == self.cow_id

    def is_matched(self):
        return self.state == TrackState.Confirmed and self._confirmed_state == ConfirmedState.Matched

    def is_reidentify(self):
        return self.state == TrackState.Confirmed and self._confirmed_state == ConfirmedState.Re_Identify

    def renew_info(self, pred, conf):
        self.predicts.append(pred)
        self.confidences.append(conf)
        if len(self.predicts)>=50:
            confidences = np.array(self.confidences)
            predicts = np.array(self.predicts)

            max_id = np.argmax(confidences)
            print('**************', predicts[max_id])
            if confidences[max_id]<0.5:
                self.cow_id='Unknown:'+str(np.round(confidences[max_id],2))
            else:
                self.cow_id=predicts[max_id]+':'+str(np.round(confidences[max_id],2))
            self.predicts = []
            self.confidences = []
            self.label = predicts[max_id]
            self._confirmed_state = ConfirmedState.Matched

    def renew_info4reid(self, pred, conf):
        self.predicts.append(pred)
        self.confidences.append(conf)
        if len(self.predicts) > 4:
            confidences = np.array(self.confidences)
            predicts = np.array(self.predicts)
            max_id = np.argmax(confidences)
            print('**************', predicts[max_id])
            if confidences[max_id] < 0.2:
                self.cow_id = 'Unknown:' + str(np.round(confidences[max_id], 2))
            # 保持一致
            else:
                self.cow_id = predicts[max_id] + ':' + str(np.round(confidences[max_id], 2))
            self.predicts = []
            self.confidences = []
            self.label = predicts[max_id]
            self._confirmed_state = ConfirmedState.Matched

    def __eq__(self, other):
        return self.track_id == other.track_id

