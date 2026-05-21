High noise prediction should be largely informed by the class. Try an all-mse run at a high batch size and see how far this can go

Test out just having momentum for the AdaLN components and embeddings for better sample efficiency


python jit/train_gsplat.py \
      --config jit/configs/jit_train_gsplat.yaml \
      --model JiT-B/8 \
      --obj_list "$DIT_GSPLAT_OBJ_LIST" --gs_path "$GS_PATH" \
      --mean_file "$MEAN_FILE" --std_file "$STD_FILE" \
      --class_map "$CLASS_MAP_PATH" --sphere2plane_path "$SPHERE2PLANE_PATH" \
      --ref_camera_tar "$REF_CAMERA_TAR" --exclude_keys_file "$EXCLUDE_KEYS_FILE" \
      --rank_transform_file "$RANK_TRANSFORM_FILE" \
      --mixed_precision bf16 \
      --results_dir output/jit_JiT-B_8_overfit103 \
      --overrides_yaml jit/configs/overrides.yaml \
      --overfit_one_per_class \
      --max_steps 3000000 --epochs 100000000 \
      --log_every 50 --val_every 500 --ckpt_every 20000 \
      --lr_warmup_steps 200 \
      --class_dropout_prob 0.0 \
      --val_cfg_scale 1.0 \
      --no-class_balanced_sampler \
      --no-wandb
