python ocr.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml --infer_img=demo/QQ20260113-204509.png -o weights=./output/model_final.pdparams --enable_ocr --ocr_lang=ch --draw_threshold=0.5


python ocr_camera.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o weights=./output/model_final.pdparams
