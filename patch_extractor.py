
from queue import Queue
from threading import Thread, Lock
import numpy as np

IS_DEBUG = True

class PatchExtractor:

    def __init__(self, dataset_low, dataset_high, label, patch_l=16, max_queue_size=4, n_workers=1, is_random=True,
                 border=None, scale=None, return_corner=False, keep_edges=True, max_N=5e4, lims_with_labels = None, patches_with_labels= 0.1, d2_after=0, two_ds=True, unlab = None):
        self.two_ds = two_ds
        self.max_queue_size = max_queue_size
        self.n_workers = n_workers
        self.d_l = dataset_low
        self.d_h = dataset_high
        self.label = label
        self.patches_with_labels = patches_with_labels
        self.lims_lab = lims_with_labels
        self.d2_after = d2_after
        self.unlab = unlab
        if IS_DEBUG:
            self.d_l1 = np.zeros_like(self.d_l)
            self.label_1 = np.zeros_like(self.label) if self.label is not None else None
        if self.label is not None:
            self.is_HR_label = not (self.d_l.shape[0:2] == self.label.shape[0:2])
        else:
            self.is_HR_label = False

        self.is_random = is_random
        self.border = border
        self.scale = scale
        self.nr_patches = max_N
        self.return_corner = return_corner
        self.keep_edges = keep_edges

        self.patch_l = patch_l
        assert self.patch_l <= self.d_l.shape[0] and self.patch_l <= self.d_l.shape[1], \
            ' patch of size {} is bigger than ds_l {}'.format(self.patch_l, self.d_l.shape)

        self.patch_h = patch_l * scale
        self.patch_lab = self.patch_h if self.is_HR_label else self.patch_l


        if self.border is not None:
            assert self.patch_l > self.border * 2
            self.border_lab = self.border * self.scale if self.is_HR_label else self.border
        if not self.is_random:
            self.compute_tile_ranges()

        else:

            n_x, self.n_y = np.subtract(self.d_l.shape[0:2], self.patch_l)
            print('Max N random patches = {}'.format(n_x*self.n_y))
            # Corner is always computed in low_res data
            max_patches = min(self.nr_patches, n_x*self.n_y)
            print('Extracted random patches = {}'.format(max_patches))



            size_label_ind = max_patches


            valid_coords = np.logical_and.reduce(~np.equal(self.label, -1), axis=2) if self.label is not None else True
            if self.is_HR_label:
                valid_input = np.logical_and.reduce(~np.equal(self.d_h, 0), axis=2)
            else:
                valid_input = np.logical_and.reduce(~np.equal(self.d_l, 0), axis=2)


            valid_coords = valid_coords & valid_input
            buffer_size = self.patch_lab

            valid_coords = np.argwhere(valid_coords[buffer_size:-buffer_size, buffer_size:-buffer_size])
            # add patch//2 to y and x coordinates to correct the cropping
            valid_coords += (buffer_size)

            if self.is_HR_label:
                valid_coords = np.array([x // self.scale for x in valid_coords])


            rand_sample = np.random.choice(len(valid_coords),min(size_label_ind,len(valid_coords)),replace=False)
            indices = valid_coords[rand_sample]

            if self.two_ds:
                self.indices1 = indices
                if self.unlab is None:
                    self.indices2 = np.random.choice(len(valid_coords), size=len(self.indices1),replace=False)
                    self.indices2 = valid_coords[self.indices2]
                else:
                    valid_coords_unlab = np.less(self.unlab[...,-1], 90)

                    buffer_size_unlab = self.patch_lab

                    valid_coords_unlab = np.argwhere(valid_coords_unlab[buffer_size_unlab:-buffer_size_unlab, buffer_size_unlab:-buffer_size_unlab])
                    # add patch//2 to y and x coordinates to correct the cropping
                    valid_coords_unlab += (buffer_size_unlab)

                    self.indices2 = np.random.choice(len(valid_coords_unlab), size=len(self.indices1),replace=False)
                    self.indices2 = valid_coords_unlab[self.indices2]
                # self.indices2 = np.random.choice(n_x * self.n_y, size=len(self.indices1), replace=False)
                print(' labeled and unlabeled data are always fed within a batch')

            else:
                self.indices1 = indices
                self.indices2 = None


            self.rand_ind = 0
        self.define_queues()

    def define_queues(self):
        self.lock = Lock()
        self.inputs_queue = Queue(maxsize=self.max_queue_size)
        self._start_batch_makers(self.n_workers)

    def _start_batch_makers(self, number_of_workers):
        for w in range(number_of_workers):
            worker = Thread(target=self._inputs_producer)
            worker.setDaemon(True)
            worker.start()

    def _inputs_producer(self):
        if self.is_random:
            while True:
                self.inputs_queue.put(self.get_random_patches())
        else:

            while True:
                with self.lock:
                    i = 0
                    # for data_id in range(len(self.d_l)):
                    for ii in self.range_i:
                        for jj in self.range_j:
                            # print(i, ii, jj)
                            if self.two_ds:
                                patches = self.get_patches(xy_corner=(ii, jj))
                                patches = np.concatenate((patches[0], patches[0]), axis=-1), np.concatenate(
                                    (patches[1], patches[1]), axis=-1)
                                self.inputs_queue.put(patches)
                            else:
                                self.inputs_queue.put(self.get_patches(xy_corner=(ii, jj)))  # + (i,)
                            i += 1
                    print('starting over Val set {}'.format(i))

    def get_patch_corner(self,data, x,y,size):
        if data is not None:
            patch = data[x:x + size, y:y+size]

            assert patch.shape == (size, size, data.shape[-1],), \
                'Shapes: Dataset={} Patch={} xy_corner={}'.format(data.shape, patch.shape, (x, y))
            assert not np.all(patch == 0.0) or not np.all(patch == -1.0)
        else:
            patch=None
        return patch
    def get_patches(self, xy_corner):

        if self.scale is None:
            scale = 1
        else:
            scale = self.scale
        x_l, y_l = map(int, xy_corner)
        x_h, y_h = x_l * scale, y_l * scale
        x_lab,y_lab = (x_h,y_h) if self.is_HR_label else (x_l,y_l)

        patch_h = self.get_patch_corner(self.d_h,x_h,y_h,self.patch_h)
        label = self.get_patch_corner(self.label,x_lab,y_lab,self.patch_lab)
        # if self.scale is not None:
        patch_l = self.get_patch_corner(self.d_l, x_l, y_l, self.patch_l)

        if IS_DEBUG and label is not None:
            self.label_1[x_lab:x_lab + self.patch_lab,
                y_lab:y_lab + self.patch_lab] = label

        if IS_DEBUG and patch_l is not None:
                self.d_l1[x_l:x_l + self.patch_l,
                      y_l:y_l + self.patch_l]=patch_l

        if self.is_HR_label:
            data_h = np.dstack((patch_h, label)) if patch_h is not None else label
        else:
            data_h = patch_h
            if label is not None:
                patch_l = np.dstack((patch_l, label))
        if self.return_corner:
            return patch_l, data_h, xy_corner
        elif data_h is not None:
            return patch_l, data_h
        else:
            return patch_l
    def get_patches_unlab(self, xy_corner):

        x_l, y_l = map(int, xy_corner)

        patch_l = self.get_patch_corner(self.unlab, x_l, y_l, self.patch_l)
        label = np.ones((self.patch_l,self.patch_l,1))*-1.0
        patch_l = np.dstack((patch_l, label))
        data_h = None
        if self.return_corner:
            return patch_l, data_h, xy_corner
        else:
            return patch_l, data_h
    def get_random_patches(self):


        if not self.two_ds:
            with self.lock:
                ind1 = self.indices1[np.mod(self.rand_ind, len(self.indices1))]

                # if self.rand_ind < self.d2_after:
                #     ind = self.indices[np.mod(self.rand_ind,self.len_labels)]
                # else:
                #     if not self.shuffled_indices:
                #         np.random.shuffle(self.indices)
                #         self.shuffled_indices = True
                #         print(' Shuffling and addind unlabeled data')
                #     ind = self.indices[np.mod(self.rand_ind,self.len_labelsALL)]

                # print ' rand_index={}'.format(self.rand_ind)
                self.rand_ind+=1
            # ind = 50
            # corner_ = divmod(ind, self.n_y)

            return self.get_patches(ind1)
        else:
            with self.lock:
                ind1 = self.indices1[np.mod(self.rand_ind, len(self.indices1))]
                ind2 = self.indices2[np.mod(self.rand_ind, len(self.indices2))]

                # print ' rand_index={}'.format(self.rand_ind)
                self.rand_ind += 1
                # ind = 50
            # corner1_ = divmod(ind1, self.n_y)
            # corner2_ = divmod(ind2, self.n_y)
            patches1 = self.get_patches(ind1)
            if self.unlab is None:
                patches2 = self.get_patches(ind2)
                patches = np.concatenate((patches1[0], patches2[0]), axis=-1), np.concatenate(
                    (patches1[1], patches2[1]), axis=-1)
                return patches
            else:
                patches2 = self.get_patches_unlab(ind2)
                return np.concatenate((patches1[0],patches2[0]),axis=-1) , np.concatenate(
                    (patches1[1], patches1[1]), axis=-1)


    def compute_tile_ranges(self):

        borders = (self.border, self.border)
        borders_h = (self.border * self.scale, self.border * self.scale)
        borders_lab = (self.border_lab, self.border_lab)
        self.range_i = [None] * len(self.d_l)
        self.range_j = [None] * len(self.d_l)
        self.nr_patches = [None] * len(self.d_l)

        # for i, data_ in enumerate(self.d_l):

        data_ = np.pad(self.d_l, (borders, borders, (0, 0)), mode='symmetric')

        range_i = np.arange(0, (data_.shape[0] - 2 * self.border) // (self.patch_l - 2 * self.border)) * (
            self.patch_l - 2 * self.border)
        range_j = np.arange(0, (data_.shape[1] - 2 * self.border) // (self.patch_l - 2 * self.border)) * (
            self.patch_l - 2 * self.border)

        x_excess = np.mod(data_.shape[0] - 2 * self.border, self.patch_l - 2 * self.border)
        y_excess = np.mod(data_.shape[1] - 2 * self.border, self.patch_l - 2 * self.border)

        if not (x_excess == 0):
            if self.keep_edges:
                range_i = np.append(range_i, (data_.shape[0] - self.patch_l))
            else:
                print('{} pixels in x axis will be discarded'.format(x_excess))
        if not (y_excess == 0):
            if self.keep_edges:
                range_j = np.append(range_j, (data_.shape[1] - self.patch_l))
            else:
                print('{} pixels in y axis will be discarded'.format(y_excess))

        # nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)
        nr_patches = len(range_i) * len(range_j)

        print('   Shapes Original = {}'.format(data_.shape))
        print('   Shapes Patched (low-res) Dataset = {}'.format(
            [nr_patches, self.patch_l, self.patch_l, data_.shape[-1]]))

        self.d_l = data_

        self.nr_patches = nr_patches
        self.range_i = range_i
        self.range_j = range_j

        self.d_h = np.pad(self.d_h, (borders_h, borders_h, (0, 0)), mode='symmetric') if self.d_h is not None else None

        self.label = np.pad(self.label, (borders_lab, borders_lab, (0, 0)),
                            mode='symmetric') if self.label is not None else None


        if IS_DEBUG:
            self.d_l1 = np.zeros_like(self.d_l)
            self.label_1 = np.zeros_like(self.label)
    def get_inputs(self):
        return self.inputs_queue.get()

    def get_iter(self):
        while True:
            yield self.get_inputs()[0:2]
    def get_iter_test(self):
        while True:
            yield self.get_inputs()


