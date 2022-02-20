import time
from options.train_options import TrainOptions
from model import create_model
from tqdm import tqdm
import os
import torch
from util.utils1 import init_dist
from dataloader.data_loader1 import CreateDataLoader
import util.data_util as data_util
import pdb


if __name__ == '__main__':
    # get training options
    trainoptions = TrainOptions()
    opt = trainoptions.parse()
    # distributed learning initiate
    if opt.dist:
        _, opt.num_gpus = init_dist()
        opt.rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        opt.rank = -1
        if opt.rank <=0:
            trainoptions.print_options(opt)

    # create a dataset
    opt.actions = data_util.define_actions(opt.actions_filter, opt.dataset_name)
    if opt.pro_train:
        train_data, train_data_loader = CreateDataLoader(opt, train=True, actions=opt.actions)
        opt.id2action = train_data.id2action
    if opt.pro_test:
        test_data, test_data_loader = CreateDataLoader(opt, train=False, actions=opt.actions)
        opt.id2action = test_data.id2action
    
    # create a model
    model = create_model(opt)


    total_iteration = 1

    # training process
    for epoch in range(1, opt.nepoch):

        torch.cuda.synchronize()
        torch.backends.cudnn.benchmark = True
        if opt.rank <= 0:
            print('======>>>>> Online epoch: #%d <<<<<======' % (epoch))
            print(opt.name)
        # for training
        if opt.pro_train:

            opt.save_img_dir = os.path.join('./saved_files/saved_imgs/', opt.name, str(epoch), 'train')
            opt.save_video_dir = os.path.join('./saved_files/saved_videos/', opt.name, str(epoch), 'train')
            epoch_start_time = time.time()
            if opt.rank <= 0:
                print('Training epoch: %d' % epoch)
            model.train()
            iteration = 1
            for i, data in enumerate(tqdm(train_data_loader, 0)):
                iter_start_time = time.time()

                model.set_input(data, epoch=epoch, i=i)
                model.optimize_parameters()



                # display images on visdom and save images
                if opt.rank <= 0:

                    # print training loss and save logging information to the disk
                    if total_iteration % opt.print_freq == 0:
                        losses = model.get_current_errors()
                        t = (time.time() - iter_start_time) / opt.batchSize
                        
                    iteration += 1
                    total_iteration += 1



            model.update_learning_rate()
            timer = time.time() - epoch_start_time
            timer = timer / len(train_data)
            if opt.rank <= 0:
                print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))
            # model.print_loss_avg()

        # for testing
        if opt.pro_test:

            opt.save_img_dir = os.path.join('./saved_files/saved_imgs/', opt.name, str(epoch), 'test')
            opt.save_video_dir = os.path.join('./saved_files/saved_videos/', opt.name, str(epoch), 'test')
            epoch_start_time = time.time()
            if opt.rank <= 0:
                print('Testing epoch: %d' % epoch)

            model.eval()
            for i, data in enumerate(tqdm(test_data_loader, 0)):
                model.set_input(data, epoch=epoch, i=i)
              
                model.test(get_all_samplings=False)
            

            timer = time.time() - epoch_start_time
            timer = timer / len(test_data)
            if opt.rank <= 0:
                print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

        # save models
        if opt.save_latest_model == 1 and opt.rank<=0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
            model.save_networks('latest')

    if opt.rank <= 0:
        print('\nEnd training')
