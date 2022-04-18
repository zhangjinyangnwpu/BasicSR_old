import torch
from collections import OrderedDict
from os import path as osp

import torchvision.utils
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.utils import SRMDPreprocessing


@MODEL_REGISTRY.register()
class PASRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(PASRModel, self).__init__(opt)
        self.opt = opt
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.scale = self.opt['network_g'].get('scale', None)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        self.contrastive_step = self.opt['train'].get('contrastive_step')
        if self.is_train:
            self.init_training_settings()

        self.degrade_flag = self.opt['degrade'].get('flag', None)
        scale = self.opt['degrade'].get('scale', None)
        mode = self.opt['degrade'].get('mode', None)
        kernel_size = self.opt['degrade'].get('kernel_size', None)
        blur_type = self.opt['degrade'].get('blur_type', None)
        sig = self.opt['degrade'].get('sig', None)
        sig_min = self.opt['degrade'].get('sig_min', None)
        sig_max = self.opt['degrade'].get('sig_max', None)
        lambda_1 = self.opt['degrade'].get('lambda_1', None)
        lambda_2 = self.opt['degrade'].get('lambda_2', None)
        theta = self.opt['degrade'].get('theta', None)
        lambda_min = self.opt['degrade'].get('lambda_min', None)
        lambda_max = self.opt['degrade'].get('lambda_max', None)
        noise = self.opt['degrade'].get('lambda_max', None)

        self.degrade = SRMDPreprocessing(
            scale=scale,
            mode = mode,
            kernel_size = kernel_size,
            blur_type=blur_type,
            sig = sig,
            sig_min=sig_min,
            sig_max=sig_max,
            lambda_1 = lambda_1,
            lambda_2 = lambda_2,
            theta = theta,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            noise=noise
        )
        self.train_or_val = 'train'

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('contrastive_opt'):
            self.cri_contrastive = build_loss(train_opt['contrastive_opt']).to(self.device)
        else:
            self.cri_contrastive = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        if self.degrade_flag:
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                # logger = get_root_logger()
                if self.train_or_val =='train':
                    self.lq,_ = self.degrade(self.gt,random=True)
                    # logger.info(f'training feed')
                elif self.train_or_val =='val':
                    self.lq,_ = self.degrade(self.gt,random=False)
                    # logger.info(f'val feed')
            # print(self.gt.shape,self.lq.shape)
            # torchvision.utils.save_image(torchvision.utils.make_grid(self.gt.detach().cpu()),'degrade_hr_bic.png')
            # torchvision.utils.save_image(torchvision.utils.make_grid(self.lq.detach().cpu()),'degrade_lr_bic.png')
            # exit()
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
            # torchvision.utils.save_image(torchvision.utils.make_grid(self.gt.detach().cpu()),'undegrade_hr1.png')
            # torchvision.utils.save_image(torchvision.utils.make_grid(self.lq.detach().cpu()),'undegrade_lr1.png')
        # exit()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # contrastive loss
        if self.cri_contrastive:
            if self.contrastive_step < current_iter:
                if self.opt['num_gpu'] > 1:
                    feature_output = self.net_g.module.get_feature(self.output)
                    feature_gt = self.net_g.module.get_feature(self.gt)
                    lq_inter = torch.nn.functional.interpolate(self.lq,scale_factor=self.scale,mode='bicubic')
                    feature_lq = self.net_g.module.get_feature(lq_inter)
                else:
                    feature_output = self.net_g.get_feature(self.output)
                    feature_gt = self.net_g.get_feature(self.gt)
                    lq_inter = torch.nn.functional.interpolate(self.lq,scale_factor=self.scale,mode='bicubic')
                    feature_lq = self.net_g.get_feature(lq_inter)
                l_contrastive = self.cri_contrastive(feature_gt,feature_output,feature_lq)
                l_total += l_contrastive
                loss_dict['l_contrastive'] = l_contrastive
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.train_or_val = 'val'
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.train_or_val = 'train'

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
