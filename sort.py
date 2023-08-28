"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)



def linear_assignment(cost_matrix):
    """
    線形割り当て問題を解く関数です。この問題は、与えられたコスト行列に基づいて最適な割り当てを見つけるものです。
    :param cost_matrix: コスト行列。各要素は割り当てのコストを示します。
    :return: 最適な割り当ての結果を示す配列。
    """
    try:
        # LAPJV (Jonker-Volgenant algorithm) を使用して線形割り当てを解くためのライブラリ。
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        # lapライブラリが存在しない場合、scipyの関数を使って線形割り当てを解きます。
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """
    複数のバウンディングボックス間のIoU (Intersection over Union) を計算します。
    IoUは、2つのバウンディングボックスの重なりの度合いを示す指標です。
    :param bb_test: テストするバウンディングボックスの配列。
    :param bb_gt: 基準となるバウンディングボックスの配列。
    :return: 各テストバウンディングボックスに対するIoUの値の配列。
    """
    # 各バウンディングボックスを2次元配列に変換します。
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    # 重なりの座標を計算します。
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    # 重なりの幅と高さを計算します。
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    # IoUを計算します。
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    バウンディングボックスの座標を、中心座標とスケール、アスペクト比の形式に変換します。
    :param bbox: [x1,y1,x2,y2]の形式のバウンディングボックスの座標。
    :return: [x,y,s,r]の形式の座標。x,yは中心座標、sは面積、rはアスペクト比。
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # 面積
    r = w / float(h)  # アスペクト比
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    中心座標とスケール、アスペクト比の形式を、バウンディングボックスの座標に変換します。
    :param x: [x,y,s,r]の形式の座標。x,yは中心座標、sは面積、rはアスペクト比。
    :param score: オプションで、バウンディングボックスの信頼度スコアを含めることができます。
    :return: [x1,y1,x2,y2]または[x1,y1,x2,y2,score]の形式のバウンディングボックスの座標。
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))


# def linear_assignment(cost_matrix):
#   try:
#     import lap
#     _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
#     return np.array([[y[i],i] for i in x if i >= 0]) #
#   except ImportError:
#     from scipy.optimize import linear_sum_assignment
#     x, y = linear_sum_assignment(cost_matrix)
#     return np.array(list(zip(x, y)))


# def iou_batch(bb_test, bb_gt):
#   """
#   From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#   """
#   bb_gt = np.expand_dims(bb_gt, 0)
#   bb_test = np.expand_dims(bb_test, 1)
  
#   xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
#   yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#   xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#   yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#   w = np.maximum(0., xx2 - xx1)
#   h = np.maximum(0., yy2 - yy1)
#   wh = w * h
#   o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
#     + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
#   return(o)  


# def convert_bbox_to_z(bbox):
#   """
#   Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#     [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#     the aspect ratio
#   """
#   w = bbox[2] - bbox[0]
#   h = bbox[3] - bbox[1]
#   x = bbox[0] + w/2.
#   y = bbox[1] + h/2.
#   s = w * h    #scale is just area
#   r = w / float(h)
#   return np.array([x, y, s, r]).reshape((4, 1))


# def convert_x_to_bbox(x,score=None):
#   """
#   Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#     [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#   """
#   w = np.sqrt(x[2] * x[3])
#   h = x[2] / w
#   if(score==None):
#     return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
#   else:
#     return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))



class KalmanBoxTracker(object):
    """
    このクラスは、バウンディングボックスとして観測される個々のトラッキングオブジェクトの内部状態を表します。
    カルマンフィルタは、ノイズのあるデータからシステムの状態を推定するための方法です。
    """

    # トラッカーの総数を追跡するための静的変数。これは、新しいトラッカーが作成されるたびにインクリメントされます。
    count = 0

    def __init__(self, bbox):
        """
        初期のバウンディングボックスを使用してトラッカーを初期化します。
        :param bbox: 初期バウンディングボックスの座標。
        """
        # 定数速度モデルのカルマンフィルタを定義します。
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 状態遷移行列 'F' を定義します。これは、一つの状態から次の状態への変化をモデル化します。
        # このモデルでは、位置と速度の両方を考慮しています。
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])

        # 観測行列 'H' を定義します。これは、実際の状態空間から観測空間へのマッピングを行います。
        # この場合、実際の位置のみが観測されます。
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        # 観測ノイズの共分散を調整します。これは、観測の不確実性を示しています。
        self.kf.R[2:, 2:] *= 10.

        # 初期の速度に対して高い不確実性を与えます。
        # 速度は初めて観測されていないため、この不確実性が存在します。
        self.kf.P[4:, 4:] *= 1000.

        # プロセスノイズの共分散行列を調整します。これは、モデルが完璧でないことを考慮したものです。
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # 提供されたバウンディングボックスを使用してフィルタの初期状態を初期化します。
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # トラッカーの状態に関するその他の変数を初期化します。
        self.time_since_update = 0  # 最後の更新からの経過時間
        self.id = KalmanBoxTracker.count  # このトラッカーのID
        KalmanBoxTracker.count += 1  # トラッカーの総数を更新
        self.history = []  # バウンディングボックスの予測の履歴
        self.hits = 0  # ディテクションマッチの回数
        self.hit_streak = 0  # 連続ディテクションマッチの回数
        self.age = 0  # トラッカーの年齢

    def update(self, bbox):
        """
        観測されたバウンディングボックスで状態ベクトルを更新します。
        """
        # 最後の更新からの経過時間をリセットします。
        self.time_since_update = 0

        # トラッカーの履歴をクリアします。
        self.history = []

        # トラッカーが正確な観測にマッチした回数をインクリメントします。
        self.hits += 1

        # 連続してマッチした回数をインクリメントします。
        self.hit_streak += 1

        # カルマンフィルタを使用して、新しい観測データで状態を更新します。
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        状態ベクトルを進め、予測されたバウンディングボックスの推定値を返します。
        """
        # もし予測された高さが0以下になる場合、高さの変化の速度を0にリセットします。
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        # カルマンフィルタを使用して次の状態を予測します。
        self.kf.predict()

        # トラッカーの「年齢」をインクリメントします。これはトラッカーが存在しているフレーム数を示します。
        self.age += 1

        # 最後の更新から時間が経過した場合、連続マッチのストリークをリセットします。
        if self.time_since_update > 0:
            self.hit_streak = 0

        # 最後の更新からの経過時間をインクリメントします。
        self.time_since_update += 1

        # 予測されたバウンディングボックスの状態を履歴に追加します。
        self.history.append(convert_x_to_bbox(self.kf.x))

        # 最新の予測されたバウンディングボックスを返します。
        return self.history[-1]

    def get_state(self):
        """
        現在のバウンディングボックスの推定値を返します。
        """
        # カルマンフィルタの現在の状態からバウンディングボックスの座標を取得して返します。
        return convert_x_to_bbox(self.kf.x)


# class KalmanBoxTracker(object):
#   """
#   This class represents the internal state of individual tracked objects observed as bbox.
#   """
#   count = 0
#   def __init__(self,bbox):
#     """
#     Initialises a tracker using initial bounding box.
#     """
#     #define constant velocity model
#     self.kf = KalmanFilter(dim_x=7, dim_z=4) 
#     self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
#     self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

#     self.kf.R[2:,2:] *= 10.
#     self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
#     self.kf.P *= 10.
#     self.kf.Q[-1,-1] *= 0.01
#     self.kf.Q[4:,4:] *= 0.01

#     self.kf.x[:4] = convert_bbox_to_z(bbox)
#     self.time_since_update = 0
#     self.id = KalmanBoxTracker.count
#     KalmanBoxTracker.count += 1
#     self.history = []
#     self.hits = 0
#     self.hit_streak = 0
#     self.age = 0

#   def update(self,bbox):
#     """
#     Updates the state vector with observed bbox.
#     """
#     self.time_since_update = 0
#     self.history = []
#     self.hits += 1
#     self.hit_streak += 1
#     self.kf.update(convert_bbox_to_z(bbox))

#   def predict(self):
#     """
#     Advances the state vector and returns the predicted bounding box estimate.
#     """
#     if((self.kf.x[6]+self.kf.x[2])<=0):
#       self.kf.x[6] *= 0.0
#     self.kf.predict()
#     self.age += 1
#     if(self.time_since_update>0):
#       self.hit_streak = 0
#     self.time_since_update += 1
#     self.history.append(convert_x_to_bbox(self.kf.x))
#     return self.history[-1]

#   def get_state(self):
#     """
#     Returns the current bounding box estimate.
#     """
#     return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    検出されたオブジェクトと追跡中のオブジェクトを関連付ける関数。
    両方ともバウンディングボックスとして表されます。

    3つのリスト（マッチング、未マッチの検出、未マッチのトラッカー）を返します。
    """
    # トラッカーが存在しない場合、すべての検出が未マッチとして返されます。
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # すべての検出とトラッカーのペアの間のIoU（Intersection over Union）を計算します。
    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        # IoUのしきい値を超えるもののみを選択します。
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 線形割当アルゴリズムを使用して最適なマッチングを見つけます。
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    # 未マッチの検出をリスト化します。
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    # 未マッチのトラッカーをリスト化します。
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # 低いIoUを持つマッチングをフィルタリングします。
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
#   """
#   Assigns detections to tracked object (both represented as bounding boxes)

#   Returns 3 lists of matches, unmatched_detections and unmatched_trackers
#   """
#   if(len(trackers)==0):
#     return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

#   iou_matrix = iou_batch(detections, trackers)

#   if min(iou_matrix.shape) > 0:
#     a = (iou_matrix > iou_threshold).astype(np.int32)
#     if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#         matched_indices = np.stack(np.where(a), axis=1)
#     else:
#       matched_indices = linear_assignment(-iou_matrix)
#   else:
#     matched_indices = np.empty(shape=(0,2))

#   unmatched_detections = []
#   for d, det in enumerate(detections):
#     if(d not in matched_indices[:,0]):
#       unmatched_detections.append(d)
#   unmatched_trackers = []
#   for t, trk in enumerate(trackers):
#     if(t not in matched_indices[:,1]):
#       unmatched_trackers.append(t)

#   #filter out matched with low IOU
#   matches = []
#   for m in matched_indices:
#     if(iou_matrix[m[0], m[1]]<iou_threshold):
#       unmatched_detections.append(m[0])
#       unmatched_trackers.append(m[1])
#     else:
#       matches.append(m.reshape(1,2))
#   if(len(matches)==0):
#     matches = np.empty((0,2),dtype=int)
#   else:
#     matches = np.concatenate(matches,axis=0)

#   return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        SORT (Simple Online and Realtime Tracking)の主要なパラメータを設定します。
        """
        # トラッカーが連続して失敗した場合に削除されるまでのフレーム数。
        self.max_age = max_age

        # トラッカーが有効と見なされるまでのヒット数。
        self.min_hits = min_hits

        # トラッキングマッチングのためのIoU (Intersection over Union)のしきい値。
        self.iou_threshold = iou_threshold

        # 現在のトラッカーのリスト。
        self.trackers = []

        # 処理されたフレームのカウント。
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        検出結果を元にトラッキングの状態を更新します。
        """
        self.frame_count += 1

        # 既存のトラッカーから位置を予測します。
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # 無効なトラッキングを削除します。
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 検出とトラッカーを関連付けます。
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # マッチしたトラッカーを関連付けられた検出で更新します。
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 未マッチの検出のために新しいトラッカーを作成して初期化します。
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)

        # トラッカーの状態を取得して結果リストに追加します。
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))  # +1 はMOTベンチマークが正のIDを要求するため

            i -= 1

            # 古いトラッカーを削除します。
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 5))


# class Sort(object):
#   def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
#     """
#     Sets key parameters for SORT
#     """
#     self.max_age = max_age
#     self.min_hits = min_hits
#     self.iou_threshold = iou_threshold
#     self.trackers = []
#     self.frame_count = 0

#   def update(self, dets=np.empty((0, 5))):
#     """
#     Params:
#       dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#     Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
#     Returns the a similar array, where the last column is the object ID.

#     NOTE: The number of objects returned may differ from the number of detections provided.
#     """
#     self.frame_count += 1
#     # get predicted locations from existing trackers.
#     trks = np.zeros((len(self.trackers), 5))
#     to_del = []
#     ret = []
#     for t, trk in enumerate(trks):
#       pos = self.trackers[t].predict()[0]
#       trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#       if np.any(np.isnan(pos)):
#         to_del.append(t)
#     trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#     for t in reversed(to_del):
#       self.trackers.pop(t)
#     matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

#     # update matched trackers with assigned detections
#     for m in matched:
#       self.trackers[m[1]].update(dets[m[0], :])

#     # create and initialise new trackers for unmatched detections
#     for i in unmatched_dets:
#         trk = KalmanBoxTracker(dets[i,:])
#         self.trackers.append(trk)
#     i = len(self.trackers)
#     for trk in reversed(self.trackers):
#         d = trk.get_state()[0]
#         if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#           ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
#         i -= 1
#         # remove dead tracklet
#         if(trk.time_since_update > self.max_age):
#           self.trackers.pop(i)
#     if(len(ret)>0):
#       return np.concatenate(ret)
#     return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

# if __name__ == '__main__':
#   # all train
#   args = parse_args()
#   display = args.display
#   phase = args.phase
#   total_time = 0.0
#   total_frames = 0
#   colours = np.random.rand(32, 3) #used only for display
#   if(display):
#     if not os.path.exists('mot_benchmark'):
#       print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
#       exit()
#     plt.ion()
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111, aspect='equal')

#   if not os.path.exists('output'):
#     os.makedirs('output')
#   pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
#   for seq_dets_fn in glob.glob(pattern):
#     mot_tracker = Sort(max_age=args.max_age, 
#                        min_hits=args.min_hits,
#                        iou_threshold=args.iou_threshold) #create instance of the SORT tracker
#     seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
#     seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
#     with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
#       print("Processing %s."%(seq))
#       for frame in range(int(seq_dets[:,0].max())):
#         frame += 1 #detection and frame numbers begin at 1
#         dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
#         dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
#         total_frames += 1

#         if(display):
#           fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
#           im =io.imread(fn)
#           ax1.imshow(im)
#           plt.title(seq + ' Tracked Targets')

#         start_time = time.time()
#         trackers = mot_tracker.update(dets)
#         cycle_time = time.time() - start_time
#         total_time += cycle_time

#         for d in trackers:
#           print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
#           if(display):
#             d = d.astype(np.int32)
#             ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

#         if(display):
#           fig.canvas.flush_events()
#           plt.draw()
#           ax1.cla()

#   print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

#   if(display):
#     print("Note: to get real runtime results run without the option: --display")

# このコードは、mainとして実行された場合にのみ動作します。
if __name__ == '__main__':
    
    # コマンドラインからの入力を解析して、必要な引数を取得します。
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    
    # 表示のための32色をランダムに生成します。
    colours = np.random.rand(32, 3)
    
    # グラフィック表示が有効な場合、描画の設定を行います。
    if(display):
        # 'mot_benchmark'ディレクトリが存在するかをチェックします。存在しない場合、エラーメッセージを表示します。
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        # 描画のためのmatplotlibの設定を行います。
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    # 出力結果を保存する'output'ディレクトリが存在しない場合、新しく作成します。
    if not os.path.exists('output'):
        os.makedirs('output')

    # MOTデータセット内の検出結果ファイルへのパスのパターンを定義します。
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    # 上記のパターンに一致するファイルを検索します。
    for seq_dets_fn in glob.glob(pattern):
        # SORTトラッカーのインスタンスを作成します。
        mot_tracker = Sort(max_age=args.max_age, 
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)
        
        # テキストファイルから検出結果を読み込みます。
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
        
        # トラッキング結果を保存するファイルを作成します。
        with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
            print("Processing %s." % (seq))
            
            # 各フレームでの検出結果に対して、SORTトラッカーを更新します。
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # 検出の形式を[x1,y1,w,h]から[x1,y1,x2,y2]に変換します。
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                # 画像を表示する設定を行います。
                if(display):
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                # トラッカーのアップデートの開始時間を記録します。
                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                # トラッキング結果を出力ファイルに書き込みます。
                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
                    
                    # 画像上にトラッキング結果を描画します。
                    if(display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))

                # 画像の表示を更新します。
                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    # トータルのトラッキング時間とフレームレートを表示します。
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    # グラフィック表示が有効の場合、実際のランタイムを得るための注意事項を表示します。
    if(display):
        print("Note: to get real runtime results run without the option: --display")
