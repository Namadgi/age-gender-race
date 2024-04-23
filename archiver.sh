torch-model-archiver -f --model-name agre --version 1.0 \
    --model-file models/mtcnn.py --serialized-file model_weights/mtcnn.pth \
    --handler handlers/agre_handler.py --export-path model_store \
    --extra-files models/mtcnn_modules.py,models/detect_face.py,model_weights/agre.onnx