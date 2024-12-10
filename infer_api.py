from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from cyclegan_infer import load_cyclegan_model, cyclegan_infer
import uvicorn

app = FastAPI()

# 在服务器启动时加载模型
model = load_cyclegan_model()


@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    # 读取上传的图片
    contents = await file.read()
    image_raw = Image.open(io.BytesIO(contents)).convert("RGB")

    # 使用模型进行处理
    processed_image = cyclegan_infer(model, image_raw)

    # 将处理后的图片转换为字节流
    img_byte_arr = io.BytesIO()
    processed_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1234)
