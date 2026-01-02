import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .latte import Latte_models
from .latte_img import LatteIMG_models
from .latte_t2v import LatteT2V
from .latte_video import LatteVIDEO_models

from torch.optim.lr_scheduler import LambdaLR

import ipdb
def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    if 'LatteIMG' in args.model:
        return LatteIMG_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras
            )
    elif 'LatteT2V' in args.model:
        pretrained_model_path = args.pretrained_model_path
        return LatteT2V.from_pretrained_2d(pretrained_model_path, subfolder="transformer", video_length=args.video_length)
    elif 'LatteVIDEO' in args.model:
        return LatteVIDEO_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras
            )
    elif 'Latte' in args.model:
        model1 =  Latte_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras
            )
            # Use 'Latte' in args.model_noise to decide whether to return model2
        if "model_noise" in args:
            if 'Latte' in args.model_noise:
                # ipdb.set_trace()
                if "max_period" in args:
                    if "condition_noise" in args:
                        model2 = Latte_models[args.model_noise](
                            input_size=args.latent_size,
                            num_classes=args.num_classes,
                            num_frames=args.num_frames,
                            learn_sigma=args.learn_sigma_noise,
                            extras=args.extras,
                            max_period=args.max_period,
                            condition_noise = args.condition_noise
                        )
                    else:
                        model2 = Latte_models[args.model_noise](
                            input_size=args.latent_size,
                            num_classes=args.num_classes,
                            num_frames=args.num_frames,
                            learn_sigma=args.learn_sigma_noise,
                            extras=args.extras,
                            max_period=args.max_period
                        )

                else:
                    if "condition_noise" in args:
                        model2 = Latte_models[args.model_noise](
                            input_size=args.latent_size,
                            num_classes=args.num_classes,
                            num_frames=args.num_frames,
                            learn_sigma=args.learn_sigma_noise,
                            extras=args.extras,
                            condition_noise = args.condition_noise
                        )
                    else:
                        model2 = Latte_models[args.model_noise](
                            input_size=args.latent_size,
                            num_classes=args.num_classes,
                            num_frames=args.num_frames,
                            learn_sigma=args.learn_sigma_noise,
                            extras=args.extras,
                        )
                # ipdb.set_trace()
                return model1, model2

        # If 'Latte' is not in args.model_noise, only return model1 (which could be None if 'Latte' wasn't in args.model)
        return model1
    else:
        raise '{} Model Not Supported!'.format(args.model)
    