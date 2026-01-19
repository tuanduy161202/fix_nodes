from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import shutil
import sys
import nodes
import gc
from comfy import model_management as mm
from comfy_extras import nodes_upscale_model
import glob
import subprocess
from typing import Optional
import re

app = FastAPI()

# Schema cho request
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry"
    width: str = "640"
    height: str = "640"
    steps: int = 15
    cfg: float = 2.0
    noise_seed: int = 221813698179903
    add_noise: str = "enable"
    start_at_step: int = 0
    end_at_step: int = 10000
    return_with_leftover_noise: str = "disable"
    error: str = None
@app.post("/generate")
async def generate_image(req: GenerateRequest):
    if req.error:
        return {"status": "error", "error": req.error}
    print("Cuda available:", torch.cuda.is_available())
    ckpt_loader = nodes.CheckpointLoaderSimple()
    # Load sẵn model abc.safetensors để tối ưu tốc độ
    print("Loading model...")
    MODEL, CLIP, VAE = ckpt_loader.load_checkpoint(ckpt_name="vxpXLHyper_v22.safetensors")
    print("Model loaded!")
    try:
        # 1. Encode text
        text_encode = nodes.CLIPTextEncode()
        pos = text_encode.encode(clip=CLIP, text=req.prompt)[0]
        neg = text_encode.encode(clip=CLIP, text=req.negative_prompt)[0]

        # 2. Tạo Latent
        latent = nodes.EmptyLatentImage().generate(width=int(req.width), height=int(req.height))[0]

        # 3. KSampler
        sampler = nodes.KSamplerAdvanced()
        with torch.no_grad():
            samples = sampler.sample(
                model=MODEL, noise_seed=req.noise_seed, steps=req.steps, 
                cfg=req.cfg, sampler_name="euler_ancestral", scheduler="normal", 
                positive=pos, negative=neg, latent_image=latent, denoise=1.0, add_noise = req.add_noise,
                start_at_step = req.start_at_step, end_at_step = req.end_at_step,
                return_with_leftover_noise = req.return_with_leftover_noise
            )[0]

            # 4. Decode & Save
            decoded = nodes.VAEDecode().decode(vae=VAE, samples=samples)[0]
            save_node = nodes.SaveImage()
            result = save_node.save_images(images=decoded, filename_prefix="Generated_images/")
        filename = str(result["ui"]["images"][0]["filename"])
        del ckpt_loader, text_encode, sampler, MODEL, CLIP, VAE, pos, neg, latent, samples, decoded
        mm.unload_all_models()
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        # Trả về tên file hoặc đường dẫn ảnh
        print('Output saved in', filename)
        return {"status": "success", "file": "output/"+filename}
    
    except Exception as e:
        del ckpt_loader, MODEL, CLIP, VAE
        mm.unload_all_models()
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

class UpscaleRequest(BaseModel):
    images_folder: str
    model: str = "RealESRGANv2-animevideo-xsx2.pth"
@app.post("/upscale")
async def upscale_image(req:UpscaleRequest):
    print(req)
    if not os.path.exists(req.images_folder):
        raise HTTPException(status_code=404, detail="Thư mục không tồn tại")
    if os.path.exists("output/Batch_Upscale"):
        shutil.rmtree("output/Batch_Upscale")
    # 1. Lấy danh sách ảnh (hỗ trợ jpg, png, webp)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(glob.glob(os.path.join(req.images_folder, ext))))
    if not image_files:
        return {"status": "error", "message": "Không tìm thấy ảnh trong thư mục"}

    # 2. Load Upscale Model một lần duy nhất để tối ưu
    upscale_loader = nodes_upscale_model.UpscaleModelLoader()
    upscale_model = upscale_loader.load_model(model_name=req.model)[0]
    
    try:
        upscaler = nodes_upscale_model.ImageUpscaleWithModel()
        image_loader = nodes.LoadImage()
        save_node = nodes.SaveImage()
        for img_path in image_files:
            print(f"Processing: {img_path}")
            img_name = os.path.basename(img_path)
            input_path = os.path.join("input", img_name)
            shutil.copy(img_path, input_path)
            img_data = image_loader.load_image(image=img_name)[0]
            with torch.no_grad():
                upscaled_img = upscaler.upscale(upscale_model=upscale_model, image=img_data)[0]
                result = save_node.save_images(images=upscaled_img, filename_prefix=f"Batch_Upscale/{img_name.split('.')[0]}")
            os.remove(input_path)
            del img_data, upscaled_img, result
            mm.soft_empty_cache()
            torch.cuda.empty_cache()
            gc.collect()
        del upscale_loader, upscale_model, upscaler, image_loader, save_node
        mm.unload_all_models() #
        mm.soft_empty_cache() #
        torch.cuda.empty_cache()
        gc.collect()
        return {"status": "success","input_dir": req.images_folder,  "output_dir": 'output/Batch_Upscale/'}
    except Exception as e:
        try:
            del upscale_loader, upscale_model, image_loader, save_node
        except:
            pass
        mm.unload_all_models() #
        mm.soft_empty_cache() #
        torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

class VideoRequest(BaseModel):
    image_folder: str      # Đường dẫn tuyệt đối tới folder ảnh (.png)
    audio_path: Optional[str] = ""       # Đường dẫn tuyệt đối tới file audio (.mp3, .wav...)
    fps: Optional[str] = "16"
    crf: Optional[str] = "10"
    filename_prefix: str

@app.post("/combine-video")
async def combine_video_sync(request: VideoRequest):
    # 1. Kiểm tra tính hợp lệ của đường dẫn
    if not os.path.isdir(request.image_folder):
        raise HTTPException(status_code=400, detail="Thư mục ảnh không tồn tại")
    audio_path = request.audio_path
    no_audio = False
    if not os.path.isfile(audio_path):
        no_audio = True

    # 2. Thiết lập tên file đầu ra (định dạng h264-mp4)
    pattern = re.compile(rf"^{re.escape(request.filename_prefix)}_(\d+)\.mp4$")
    max_idx = 0
    
    # Kiểm tra nếu thư mục không tồn tại thì tạo mới và trả về 1
    if not os.path.exists('output'):
        os.makedirs('output')
    else:
        for filename in os.listdir('output'):
            match = pattern.match(filename)
            if match:
                current_idx = int(match.group(1))
                if current_idx > max_idx:
                    max_idx = current_idx
    output_file = os.path.join('output', f"{request.filename_prefix}_{(max_idx+1):05d}.mp4")

    # 3. Lệnh FFmpeg chuẩn video/h264-mp4
    # -y: Ghi đè nếu file tồn tại
    # -pattern_type glob: Nạp toàn bộ ảnh .png trong folder theo thứ tự tên
    # -pix_fmt yuv420p: Đảm bảo tương thích mọi trình xem video
    command = [
        "ffmpeg", "-y",
        "-framerate", request.fps,
        "-pattern_type", "glob", "-i", f"{request.image_folder}/*.png"]
    if not no_audio:
        command += ["-i", audio_path]
    command += [
        "-c:v", "libx264",       # Codec H.264
        "-crf", request.crf, # Chất lượng (18 là rất nét)
        "-pix_fmt", "yuv420p",   # Màu sắc chuẩn mp4
        "-c:a", "aac",           # Codec âm thanh chuẩn
        "-b:a", "192k",
        "-shortest",             # Cắt video/nhạc theo cái ngắn hơn
        output_file
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return {
            "status": "success",
            "output_video": output_file,
            "message": "Video đã được tạo thành công."
        }
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Lỗi FFmpeg: {e.stderr}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("endpoints:app", host="0.0.0.0", port=8000, reload=True)