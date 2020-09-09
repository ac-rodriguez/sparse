
from queue import Queue, Empty
from threading import Thread, Lock, Event
import threading
import numpy as np

import sys 


IS_DEBUG = True


class BasePatch:
    def __init__(self, name='BasePatch', max_queue_size=4,n_workers=1):

        self.max_queue_size=max_queue_size
        self.n_workers = n_workers
        self.name = name

    def define_queues(self):
        self.lock = Lock()
        self.inputs_queue = Queue(maxsize=self.max_queue_size)
        self._start_batch_makers(self.n_workers)

    def _start_batch_makers(self, number_of_workers):
        self.stop_event = Event()
        self.workers = []
        for w in range(number_of_workers):
            worker = Thread(target=self._inputs_producer,
             name=f'worker{w} of {self.name}')
            worker.setDaemon(True)
            worker.start()
            self.workers.append(worker)

    def finish_threads(self):
        
        self.stop_event.set()
        def consumer_():
            while True:
                out_ = self.inputs_queue.get()
                self.inputs_queue.task_done()
                if out_ is None:
                    break

        thread = Thread(target=consumer_, name='consumer')
        thread.setDaemon(True)
        thread.start()

        for w in self.workers:
            w.join()
        self.inputs_queue.put(None)

        thread.join()

    def redefine_queues(self):
        threads_ =  threading.active_count()
        if hasattr(self,'inputs_queue'):
            self.finish_threads()
        
        self.define_queues()
        print(f"Active threads after redefine: {threading.active_count()} (was {threads_})")

    def _inputs_producer(self):
        raise NotImplementedError

    def get_inputs(self):
        try:
            input = self.inputs_queue.get(timeout=30)
        except Empty:
            #print('error in queue',self.name)
            raise Empty('error in queue',self.name)
        return input

    def get_iter(self):
        while True:
            yield self.get_inputs()

    def get_iter_test(self):
        while True:
            yield self.get_inputs()

import random
class PatchWraper(BasePatch):
    def __init__(self, extractors,max_queue_size=4,n_workers=1):
        self.add_info = True
        self.extractors = [x for x in extractors if x.any_valid_pixels]
        super(PatchWraper, self).__init__('Patch_wraper',max_queue_size,n_workers)

        self.nr_patches = [len(x.indices1) for x in self.extractors]
        self.n_pixels = [(x.d_l.shape[0]*x.d_l.shape[1]) for x in self.extractors]
        self.is_onlyLR = all([x.is_onlyLR for x in self.extractors])
        #self.weights = self.n_pixels / np.sum(self.n_pixels)
        #print(f' Fetch Prob. {self.weights}')
        #self.indices1 = np.random.choice(len(self.extractors),size=100,p=self.weights)
        self.indices1 = [(id, id1) for id, ext in enumerate(self.extractors) for id1 in range(len(ext.indices1))]
        random.shuffle(self.indices1)
        self.rand_ind = 0
        self.define_queues()

    def _inputs_producer(self):
        while not self.stop_event.is_set():
            with self.lock:
                index = np.mod(self.rand_ind, len(self))
                self.rand_ind +=1
            patches = self.__getitem__(index)
            # patches = self.extractors[ind1].get_inputs()
            # print(f' dataset {ind1} fetched')
            self.inputs_queue.put(patches)
    def __len__(self):
        return len(self.indices1)

    def __getitem__(self,index):

        extr_id , index_patch = self.indices1[index]

        patches = self.extractors[extr_id].__getitem__(index_patch)
        if self.add_info:
            patches['info'] = self.extractors[extr_id].name
        return patches
    
        # is_correct = False
        # while not is_correct:
        #     patches = self.get_random_patches()
        #     is_correct =  not self.is_clouded(patches)
        # self.inputs_queue.put(patches)

class PatchExtractor(BasePatch):

    def __init__(self, dataset_low, dataset_high, label, patch_l=16, max_queue_size=4, n_workers=1, is_random=True,
                 border=None, scale=None, return_corner=False, keep_edges=True, max_N=5e4, lims_with_labels = None,
                  patches_with_labels= 0.1, d2_after=0, two_ds=True, unlab = None, use_location=False, is_use_queues=True,
                  ds_info = None):
        self.two_ds = two_ds
        assert not two_ds
        self.d_l = dataset_low
        self.is_use_queues = is_use_queues
        if ds_info is None:
            name = f'PatchExtractor({self.d_l.shape})'
        else:
            name = ds_info['tilename']+'_'+ds_info['gt'].split('/')[-1]
        super(PatchExtractor, self).__init__(name,max_queue_size,n_workers)

        self.d_h = dataset_high
        self.label = label
        self.patches_with_labels = patches_with_labels
        self.lims_lab = lims_with_labels
        self.verbose = False
        # self.d2_after = d2_after
        self.unlab = unlab
        if IS_DEBUG:
            self.d_l1 = np.zeros_like(self.d_l)
            self.label_1 = np.ones_like(self.label)*-1.0 if self.label is not None else None
        if self.label is not None:
            self.is_HR_label = not (self.d_l.shape[0:2] == self.label.shape[0:2])
        else:
            self.is_HR_label = False
        self.is_onlyLR = not self.is_HR_label and self.d_h is None
        self.is_random = is_random
        self.border = border
        self.scale = scale
        self.nr_patches = max_N
        self.return_corner = return_corner
        self.keep_edges = keep_edges
        self.clouded_patches = []

        self.is_use_location = use_location
        if self.is_use_location:
            self.cloud_axis = -3
        else:
            self.cloud_axis= -1

        self.patch_l = patch_l

        self.patch_h = patch_l * scale
        self.patch_lab = self.patch_h if self.is_HR_label else self.patch_l
        self.any_valid_pixels = True

        if not self.patch_l <= self.d_l.shape[0] and self.patch_l <= self.d_l.shape[1]:
            print('patch of size {} is bigger than ds_l {}'.format(self.patch_l, self.d_l.shape))
            print('will skip during training')
            self.any_valid_pixels = False

        if self.border is not None:
            assert self.patch_l > self.border * 2
            self.border_lab = self.border * self.scale if self.is_HR_label else self.border
        if not self.is_random:
            self.compute_tile_ranges()
            self.indices1 = [(ii,jj) for ii in self.range_i for jj in self.range_j]
            self.rand_ind = 0
        elif self.any_valid_pixels:

            n_x, self.n_y = np.subtract(self.d_l.shape[0:2], self.patch_l)
            if self.verbose: print('Max N random patches = {}'.format(n_x*self.n_y))
            # Corner is always computed in low_res data
            max_patches = min(self.nr_patches, n_x*self.n_y)
            if self.verbose: print('Extracted random patches = {}'.format(max_patches))


            size_label_ind = max_patches

            cloud_threshold = 50
            if self.label is not None:
                valid_pixels = np.logical_and.reduce(~np.equal(self.label, -1), axis=2)
            else:
                valid_pixels = True
            if self.is_HR_label:
                valid_input = np.logical_and.reduce(~np.equal(self.d_h, 0), axis=2)
            else:
                if self.d_l.shape[-1] > 3:
                    valid_input = np.logical_and.reduce(~np.equal(self.d_l[...,:self.cloud_axis], 0), axis=2)
                    valid_input = valid_input & np.less(self.d_l[...,self.cloud_axis], cloud_threshold)
                else:
                    valid_input = np.logical_and.reduce(~np.equal(self.d_l, 0), axis=2)

            valid_pixels = valid_pixels & valid_input
            buffer_size = self.patch_lab
            self.any_valid_pixels = np.any(valid_pixels[buffer_size:-buffer_size, buffer_size:-buffer_size])
            if not self.any_valid_pixels:
                print(f'no pixel is valid in the dataset with shape {valid_pixels.shape}, will skip in training')
            else:
                valid_coords = np.argwhere(valid_pixels[buffer_size:-buffer_size, buffer_size:-buffer_size])
                # add patch//2 to y and x coordinates to correct the cropping
                valid_coords += (buffer_size)

                if self.is_HR_label:
                    valid_coords = np.array([x // self.scale for x in valid_coords])


                rand_sample = np.random.choice(len(valid_coords),min(size_label_ind,len(valid_coords)),replace=False)
                indices = valid_coords[rand_sample]
                assert len(indices) > 0, f'error rand_sample {rand_sample}, valid coords {valid_coords}'

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
                    if self.verbose: print(' labeled and unlabeled data are always fed within a batch')

                else:
                    self.indices1 = indices
                    self.indices2 = None


                self.rand_ind = 0
        if self.any_valid_pixels and self.is_use_queues:
            self.define_queues()
    def __len__(self):
        return len(self.indices1)

    def __getitem__(self, index):

        patch_id = self.indices1[index]

        return self.get_patches(patch_id)


    def is_clouded(self,patches):
        # Compute if 50% of the pixels have 50% cloud prob
        cloud = patches['feat_l'][...,self.cloud_axis]
        return np.nanmean(cloud > 50) > 0.5

    def _inputs_producer(self):
        if self.is_random:
            while not self.stop_event.is_set():
                is_correct = False
                while not is_correct:
                    patches = self.get_random_patches()
                    is_correct =  not self.is_clouded(patches)
                self.inputs_queue.put(patches)
        else:
            while not self.stop_event.is_set():
                with self.lock:
                    index = np.mod(self.rand_ind, len(self))
                    self.rand_ind +=1
                    patches = self.__getitem__(index)
                    self.inputs_queue.put(patches)


            # while not self.stop_event.is_set():
            #     with self.lock:
            #         i = 0
            #         # for data_id in range(len(self.d_l)):
            #         for ii in self.range_i:
            #             for jj in self.range_j:
            #                 # print(i, ii, jj)
            #                 # if self.two_ds:
            #                 patches = self.get_patches(xy_corner=(ii, jj))
            #                     # patches, patches_h = self.get_patches(xy_corner=(ii, jj))
            #                     # if self.is_onlyLR:
            #                     #     patches = np.concatenate((patches, patches), axis=-1)
            #                     # else:
            #                     #     patches = np.concatenate((patches, patches), axis=-1), np.concatenate(
            #                     #         (patches_h, patches_h), axis=-1)
            #                 self.inputs_queue.put(patches)
            #                 # else:
            #                     # patches, patches_h = self.get_patches(xy_corner=(ii, jj))
            #                     # if not self.is_onlyLR:
            #                         # patches = (patches,patches_h)
            #                     # self.inputs_queue.put(patches)  # + (i,)
            #                 i += 1
            #         # print('starting over Val set {}'.format(i))

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
        return_dict = {}
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

        return_dict['feat_l'] = patch_l
        if label is not None:
            return_dict['label'] = label

        if patch_h is not None:
            return_dict['feat_h'] = patch_h

        if IS_DEBUG and label is not None:
            self.label_1[x_lab:x_lab + self.patch_lab,
                y_lab:y_lab + self.patch_lab] = label

        if IS_DEBUG and patch_l is not None:
                self.d_l1[x_l:x_l + self.patch_l,
                      y_l:y_l + self.patch_l]=patch_l
        return return_dict

    # def get_patches(self, xy_corner):

    #     if self.scale is None:
    #         scale = 1
    #     else:
    #         scale = self.scale
    #     x_l, y_l = map(int, xy_corner)
    #     x_h, y_h = x_l * scale, y_l * scale
    #     x_lab,y_lab = (x_h,y_h) if self.is_HR_label else (x_l,y_l)

    #     patch_h = self.get_patch_corner(self.d_h,x_h,y_h,self.patch_h)
    #     label = self.get_patch_corner(self.label,x_lab,y_lab,self.patch_lab)
    #     # if self.scale is not None:
    #     patch_l = self.get_patch_corner(self.d_l, x_l, y_l, self.patch_l)

    #     if IS_DEBUG and label is not None:
    #         self.label_1[x_lab:x_lab + self.patch_lab,
    #             y_lab:y_lab + self.patch_lab] = label

    #     if IS_DEBUG and patch_l is not None:
    #             self.d_l1[x_l:x_l + self.patch_l,
    #                   y_l:y_l + self.patch_l]=patch_l

    #     if self.is_HR_label:
    #         data_h = np.dstack((patch_h, label)) if patch_h is not None else label
    #     else:
    #         data_h = patch_h
    #         if label is not None:
    #             patch_l = np.dstack((patch_l, label))
    #     if self.return_corner:
    #         return patch_l, data_h, xy_corner
    #     else:
    #         return patch_l, data_h

    def get_patches_unlab(self, xy_corner):
        raise NotImplementedError

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
                self.rand_ind+=1

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
            return_dict = self.get_patches(ind1)
            if self.unlab is None:
                return_dict1 = self.get_patches(ind2)
                for key, val in return_dict1.items():
                    if key != 'label':
                        return_dict[key+'U'] = val
            else:
                raise NotImplementedError
            return return_dict

    # def get_random_patches(self):

    #     if not self.two_ds:
    #         with self.lock:
    #             ind1 = self.indices1[np.mod(self.rand_ind, len(self.indices1))]
    #             self.rand_ind+=1

    #         return self.get_patches(ind1)
    #     else:
    #         with self.lock:
    #             ind1 = self.indices1[np.mod(self.rand_ind, len(self.indices1))]
    #             ind2 = self.indices2[np.mod(self.rand_ind, len(self.indices2))]

    #             # print ' rand_index={}'.format(self.rand_ind)
    #             self.rand_ind += 1
    #             # ind = 50
    #         # corner1_ = divmod(ind1, self.n_y)
    #         # corner2_ = divmod(ind2, self.n_y)
    #         patches1, patches1_h = self.get_patches(ind1)
    #         if self.unlab is None:
    #             patches2, patches2_h = self.get_patches(ind2)
    #         else:
    #             patches2, patches2_h = self.get_patches_unlab(ind2)
    #         if self.is_onlyLR:
    #             patches = np.concatenate((patches1, patches2), axis=-1)
    #         else:
    #             patches = np.concatenate((patches1, patches2), axis=-1), np.concatenate(
    #                 (patches1_h, patches2), axis=-1)
    #         return patches


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
                if self.verbose: print('{} pixels in x axis will be discarded'.format(x_excess))
        if not (y_excess == 0):
            if self.keep_edges:
                range_j = np.append(range_j, (data_.shape[1] - self.patch_l))
            else:
                if self.verbose: print('{} pixels in y axis will be discarded'.format(y_excess))

        # nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)
        nr_patches = len(range_i) * len(range_j)

        if self.verbose: print('   Shapes Original = {}'.format(data_.shape))
        if self.verbose: print('   Shapes Patched (low-res) Dataset = {} (border = {})'.format(
            [nr_patches, self.patch_l, self.patch_l, data_.shape[-1]],self.border))

        if self.verbose: print(range_i)
        if self.verbose: print(range_j)
        self.d_l = data_

        self.nr_patches = nr_patches
        self.range_i = range_i
        self.range_j = range_j

        self.d_h = np.pad(self.d_h, (borders_h, borders_h, (0, 0)), mode='symmetric') if self.d_h is not None else None

        self.label = np.pad(self.label, (borders_lab, borders_lab, (0, 0)),
                            mode='symmetric') if self.label is not None else None


        if IS_DEBUG:
            self.d_l1 = np.zeros_like(self.d_l)
            self.label_1 = np.ones_like(self.label)*-1.0
    def get_inputs(self):
        return self.inputs_queue.get()

    def get_iter(self):
        while True:
            yield self.get_inputs() #[0:2]
    def get_iter_test(self):
        while True:
            yield self.get_inputs()


# https://stackoverflow.com/questions/3033952/threading-pool-similar-to-the-multiprocessing-pool
# sample code for creating a worker pool 
from queue import Queue
from threading import Thread

class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()
    
    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try: func(*args, **kargs)
            except Exception as e: print(e)
            self.tasks.task_done()

class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads): Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()

if __name__ == '__main__':
    from random import randrange
    delays = [randrange(1, 10) for i in range(100)]
    
    from time import sleep
    def wait_delay(d):
        print('sleeping for (%d)sec' % d)
        sleep(d)
    
    # 1) Init a Thread pool with the desired number of threads
    pool = ThreadPool(20)
    
    for i, d in enumerate(delays):
        # print the percentage of tasks placed in the queue
        print('%.2f%c' % ((float(i)/float(len(delays)))*100.0,'%'))
        
        # 2) Add the task to the queue
        pool.add_task(wait_delay, d)
    

    # 3) Wait for completion
    pool.wait_completion()