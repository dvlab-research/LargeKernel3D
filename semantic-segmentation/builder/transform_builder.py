# from util import transform as t
from util import transform_v2
from util import transform_me as t

def build_transform(args):
    train_transform = None
    if args.aug:
        aug_type = args.get("aug_type", "all")
        if aug_type == 'rotate':
            train_transform = transform_v2.RandomRotate(along_z=args.get('rotate_along_z', True))
        elif aug_type == 'scale':
            train_transform = transform_v2.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2))
        elif aug_type == 'jitter':

            jitter_sigma = args.get('jitter_sigma', 0.01)
            jitter_clip = args.get('jitter_clip', 0.05)

            train_transform = transform_v2.RandomJitter(sigma=jitter_sigma, clip=jitter_clip)
        elif aug_type == 'drop_color':
            train_transform = transform_v2.RandomDropColor(color_augment=args.get('color_augment', 0.0))
        elif aug_type == "no_jitter":
            train_transform = transform_v2.Compose([
                transform_v2.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform_v2.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                # transform_v2.RandomJitter(sigma=args.get('jitter_sigma', 0.01), clip=args.get('jitter_clip', 0.05)),
                transform_v2.RandomDropColor(color_augment=args.get('color_augment', 0.0))
            ])
        elif aug_type == 'all':

            jitter_sigma = args.get('jitter_sigma', 0.01)
            jitter_clip = args.get('jitter_clip', 0.05)
            
            train_transform = transform_v2.Compose([
                transform_v2.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform_v2.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                transform_v2.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
                transform_v2.RandomDropColor(color_augment=args.get('color_augment', 0.0))
            ])
        elif aug_type == 'elastic':
            
            elastic_distortion_params = args.get("elastic_distortion_params", [[0.2, 0.4], [0.8, 1.6]])

            train_transform = transform_v2.Compose([
                transform_v2.ElasticDistortion(elastic_distortion_params)
            ])
        elif aug_type == 'horizontal_flip':

            train_transform = transform_v2.Compose([
                transform_v2.RandomHorizontalFlip()
            ])

        elif aug_type == 'chromatic_auto_contrast':
            
            train_transform = transform_v2.Compose([
                transform_v2.ChromaticAutoContrast()
            ])
        
        elif aug_type == 'chromatic_translation':
            
            data_aug_color_trans_ratio = args.get("data_aug_color_trans_ratio", 0.1)

            train_transform = transform_v2.Compose([
                transform_v2.ChromaticTranslation(data_aug_color_trans_ratio)
            ])
        
        elif aug_type == 'chromatic_jitter':
            
            data_aug_color_jitter_std = args.get("data_aug_color_jitter_std", 0.05)

            train_transform = transform_v2.Compose([
                transform_v2.ChromaticJitter(data_aug_color_jitter_std)
            ])
        
        elif aug_type == 'hue_saturation_translation':
            
            data_aug_hue_max = args.get("data_aug_hue_max", 0.5)
            data_aug_saturation_max = args.get("data_aug_saturation_max", 0.2)

            train_transform = transform_v2.Compose([
                transform_v2.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max)
            ])
        
        elif aug_type == 'scale+rotate+elastic+horizontal_flip':
            
            elastic_distortion_params = args.get("elastic_distortion_params", [[0.2, 0.4], [0.8, 1.6]])

            train_transform = transform_v2.Compose([
                transform_v2.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                transform_v2.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform_v2.ElasticDistortion(elastic_distortion_params),
                transform_v2.RandomHorizontalFlip()
            ])

        elif aug_type == 'scale+rotate+elastic+horizontal_flip+drop_color':
            
            elastic_distortion_params = args.get("elastic_distortion_params", [[0.2, 0.4], [0.8, 1.6]])

            train_transform = transform_v2.Compose([
                transform_v2.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                transform_v2.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform_v2.ElasticDistortion(elastic_distortion_params),
                transform_v2.RandomHorizontalFlip(),
                transform_v2.RandomDropColor(color_augment=args.get('color_augment', 0.0))
            ])

        else:
            raise ValueError("No such aug type: {}".format(aug_type))
    
    return train_transform

def build_transform_me(args):
    elastic_distort_params = ((0.2, 0.4), (0.8, 1.6))
    
    prevoxel_transform_train = []
    if (args.aug and args.get('distill_mode', None)==None) or args.get('pre_aug', False):
        prevoxel_transform_train.append(t.ElasticDistortion(elastic_distort_params))
    if len(prevoxel_transform_train) > 0:
        prevoxel_transform_train = t.Compose(prevoxel_transform_train)
    else:
        prevoxel_transform_train = None
    
    # args.drop_align = args.get("drop_align", False)
    train_transform = []
    if args.aug:
        if args.get('distill_mode', None):
            train_transform += [
                t.RandomDropoutDitill(0.2, use_ori_coords=args.get("use_ori_coords", False)),
                t.RandomHorizontalFlipDistill(args.rotation_axis, args.is_temporal, use_ori_coords=args.get("use_ori_coords", False)),
                t.ChromaticAutoContrastDistill(use_ori_coords=args.get("use_ori_coords", False)),
                t.ChromaticTranslationDistill(args.data_aug_color_trans_ratio, use_ori_coords=args.get("use_ori_coords", False)),
                t.ChromaticJitterDistill(args.data_aug_color_jitter_std, use_ori_coords=args.get("use_ori_coords", False)),
                # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
            ]
        else:
            train_transform += [
                t.RandomDropout(0.2, use_ori_coords=args.get("use_ori_coords", False)),
                t.RandomHorizontalFlip(args.rotation_axis, args.is_temporal, use_ori_coords=args.get("use_ori_coords", False)),
                t.ChromaticAutoContrast(use_ori_coords=args.get("use_ori_coords", False)),
                t.ChromaticTranslation(args.data_aug_color_trans_ratio, use_ori_coords=args.get("use_ori_coords", False)),
                t.ChromaticJitter(args.data_aug_color_jitter_std, use_ori_coords=args.get("use_ori_coords", False)),
                # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
            ]

    if len(train_transform) > 0:
        train_transform = t.Compose(train_transform)
    else:
        train_transform = None

    return prevoxel_transform_train, train_transform