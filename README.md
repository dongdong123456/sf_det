训练
python tools/train.py -f exps/example/custom/yolox_s.py -d 0 -b 8 --fp16 -c yolox_s.pth
python tools/train.py -f exps/example/custom/yolox_tiny.py -d 0 -b 32 --fp16 -c yolox_tiny.pth
python tools/train.py -f exps/example/custom/yolox_s.py -d 0 -b 8 --fp16 -c best_ckpt.pth

计算AP
python tools/eval.py -n  yolox-s -c YOLOX_outputs/yolox_s/best_ckpt.pth -b 4 -d 0 --conf 0.001 [--fp16] [--fuse]

测试
python tools/demo.py image -n yolox-s -c YOLOX_outputs/yolox_s/best_ckpt.pth --path assets/1.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]


点击start.bat 就可以
