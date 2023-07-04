import gradio as gr
import numpy as np
from diffusers import StableDiffusionPipeline, DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import torch
import PIL.Image
import datetime

# Check environment
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

device = "cuda"

schedulers = {
    "DDPMScheduler": DDPMScheduler,
    "DDIMScheduler": DDIMScheduler,
    "PNDMScheduler": PNDMScheduler,
    "LMSDiscreteScheduler": LMSDiscreteScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler
}

class Model:
    def __init__(self, modelID, schedulerName):
        self.modelID = modelID
        self.pipe = StableDiffusionPipeline.from_pretrained(modelID, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.scheduler = schedulers[schedulerName].from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    def process(self, 
                prompt: str, 
                negative_prompt: str,
                guidance_scale:int = 6,
                num_images:int = 1,
                num_steps:int = 35):
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        generator = torch.Generator(device).manual_seed(seed)
        now = datetime.datetime.now()
        print(now)
        print(self.modelID)
        print(prompt)
        print(negative_prompt)
        with torch.inference_mode():
            images = self.pipe(prompt=prompt,
                         negative_prompt=negative_prompt,
                         guidance_scale=guidance_scale,
                         num_images_per_prompt=num_images,
                         num_inference_steps=num_steps,
                         generator=generator,
                         height=768, 
                         width=768).images  
        images = [PIL.Image.fromarray(np.array(img)) for img in images]
        return images




                    
def generateImage(prompt, n_prompt, modelName, schedulerName):
    images = models[modelName].process(prompt, n_prompt)
    images = [np.array(img) for img in images]
    return images[0]  # Return the first image

def create_demo():
    # Settings are defined here
    prompt = gr.inputs.Textbox(label='Prompt',default='a sprinkled donut sitting on top of a table, blender donut tutorial, colorful hyperrealism, everything is made of candy, hyperrealistic digital painting, covered in sprinkles and crumbs, vibrant colors hyper realism, colorful smoke explosion background')
    n_prompt = gr.inputs.Textbox(
        label='Negative Prompt',
        default='(disfigured), ((bad art)), ((deformed)), ((extra limbs)), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, poorly drawn eyes, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), cloned face, body out of frame, out of frame, bad anatomy, gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), (((long neck))), Deformed, blurry'
    )
    modelName = gr.inputs.Dropdown(choices=list(models.keys()), 
                                   label="FFusion Test Model",
                                   default=list(models.keys())[0])  # Set the default model
    schedulerName = gr.inputs.Dropdown(choices=list(schedulers.keys()), 
                                       label="Scheduler",
                                       default=list(schedulers.keys())[0])  # Set the default scheduler
    inputs = [prompt, n_prompt, modelName, schedulerName]

    # Images are displayed here
    result = gr.outputs.Image(label='Output', type="numpy")

    # Define the function to run when the button is clicked
    def run(prompt, n_prompt, modelName, schedulerName):
        return generateImage(prompt, n_prompt, modelName, schedulerName)

    # Create the interface
    iface = gr.Interface(
    fn=run,
    inputs=inputs,
    outputs=result,
    layout=[
        gr.Markdown("### FFusion.AI - beta Playground"),
        inputs,
        result
    ]
)

    return iface

if __name__ == '__main__':
    models = {
        "FFUSION.ai-768-BaSE": Model("FFusion/FFusion-BaSE", list(schedulers.keys())[0]),
        "FFUSION.ai-v2.1-768-BaSE-alpha-preview": Model("FFusion/di.FFUSION.ai-v2.1-768-BaSE-alpha", list(schedulers.keys())[0]),
    "FFusion.ai.Beta-512": Model("FFusion/di.ffusion.ai.Beta512", list(schedulers.keys())[0])
    }
    demo = create_demo()
    demo.launch()
