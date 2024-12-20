import os
import uuid
import logging
import requests
import websocket
import json
import time
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)

class ComfyWorker:
    def __init__(self, api_base: str = None, api_key: str = None, comfy_url: str = None):
        logger.info("Initializing ComfyWorker...")
        
        # API settings
        self.api_base = api_base or os.getenv('API_BASE_URL', 'http://localhost:3000/api/v1')
        self.api_key = api_key or os.getenv('API_KEY')
        self.headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        
        # ComfyUI settings
        self.comfy_url = comfy_url or os.getenv('COMFY_URL', 'http://127.0.0.1:8188')
        
        # Worker ID
        self.worker_id = str(uuid.uuid4())
        
        # Task tracking
        self.current_task = None
        
        logger.info(f"Worker initialized with API base URL: {self.api_base}")
        logger.info(f"ComfyUI URL: {self.comfy_url}")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            if data['type'] == 'executing':
                logger.info(f"Executing node: {data['data'].get('node', None)}")
                
            elif data['type'] == 'progress':
                logger.info(f"Progress: {data['data']['value']}/{data['data']['max']}")
                
            elif data['type'] == 'executed':
                logger.info(f"Executed: {data['data']}")
                if data['data']['node'] is None:  # Workflow completed
                    logger.info("Workflow execution completed")
                    self.handle_workflow_completion()
                    
            elif data['type'] == 'status':
                logger.info(f"Status: {data['data']}")

        except Exception as e:
            logger.error(f"Error processing websocket message: {str(e)}", exc_info=True)

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {str(error)}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a prompt in ComfyUI"""
        response = requests.post(
            f"{self.comfy_url}/prompt",
            json={"prompt": workflow}
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to queue prompt: {response.status_code}")
        return response.json()['prompt_id']

    def get_image(self, filename: str) -> Image.Image:
        """Get an image from ComfyUI's output"""
        response = requests.get(f"{self.comfy_url}/view?filename={filename}")
        if response.status_code != 200:
            raise ValueError(f"Failed to get image: {response.status_code}")
        return Image.open(io.BytesIO(response.content))

    def handle_workflow_completion(self):
        """Handle workflow completion and update task status"""
        if not self.current_task:
            return

        try:
            # Get the latest history
            response = requests.get(f"{self.comfy_url}/history")
            if response.status_code != 200:
                raise ValueError("Failed to get history")

            history = response.json()
            latest_output = list(history.values())[-1]
            
            # Get output image
            output_images = []
            for node_id, node_output in latest_output.get('outputs', {}).items():
                if 'images' in node_output:
                    for image_data in node_output['images']:
                        output_images.append(image_data['filename'])

            if output_images:
                # Save and upload the first image
                image = self.get_image(output_images[0])
                output_path = f"outputs/{self.current_task['id']}.png"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
                
                # Upload image
                image_url = self.upload_file(output_path, self.current_task['id'])
                logger.info(f"Image uploaded successfully: {image_url}")
                
                # Update task status
                self.update_task(
                    self.current_task['id'],
                    status="success",
                    output={
                        "image_urls": [image_url]
                    }
                )
                
                # Cleanup
                os.remove(output_path)
                
            self.current_task = None
            
        except Exception as e:
            logger.error(f"Error handling workflow completion: {str(e)}", exc_info=True)
            if self.current_task:
                self.update_task(
                    self.current_task['id'],
                    status="failed",
                    error={
                        "code": 10001,
                        "message": str(e)
                    }
                )
                self.current_task = None
    
    def get_workflow_path(self, _: dict) -> str:
        return "workflows/flux_dev_q8.json"

    def execute_workflow(self, workflow: dict) -> list:
        """Execute workflow and wait for results"""
        # Setup WebSocket
        ws_url = self.comfy_url.replace('http', 'ws') + '/ws'
        ws = websocket.create_connection(ws_url)
        try:
            # Queue the prompt
            prompt_id = self.queue_prompt(workflow)
            logger.info(f"Workflow queued with prompt_id: {prompt_id}")
            
            # Wait for execution to complete
            task_started = False
            current_progress_percentage = 0

            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    
                    if message['type'] == 'executing':
                        data = message['data']
                        logger.info(f"Executing node: {data.get('node', None)}")
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            logger.info("Workflow execution completed")
                            break
                            
                    elif message['type'] == 'progress':
                        data = message['data']
                        logger.info(f"Progress: {data['value']}/{data['max']}")
                        # get the progress percentage as integer
                        progress_percentage = int(data['value'] / data['max'] * 100)
                        # update the task with the progress percentage if it is higher than the current progress percentage
                        if progress_percentage > current_progress_percentage:
                            self.update_task(
                                self.current_task['id'],
                                status="processing",
                                progress=progress_percentage
                            )
                            current_progress_percentage = progress_percentage
                            logger.info(f"Progress updated: {progress_percentage}%")
                        
                    elif message['type'] == 'status':
                        logger.info(f"Status: {message['data']}")
                        if message['data']['status']['exec_info']['queue_remaining'] == 0:
                            if task_started:
                                logger.info("Workflow execution completed")
                                break
                            else:
                                task_started = True
                        
                else:
                    # Binary data (preview images) - skip for now
                    continue

            # Get results from history
            response = requests.get(f"{self.comfy_url}/history")
            if response.status_code != 200:
                raise ValueError("Failed to get history")

            history = response.json()[prompt_id]
            output_images = []
            
            # Collect all output images
            for node_id, node_output in history.get('outputs', {}).items():
                if 'images' in node_output:
                    for image_data in node_output['images']:
                        image = self.get_image(image_data['filename'])
                        output_images.append(image)

            return output_images

        finally:
            # Always close the WebSocket
            ws.close()

    def process_task(self, task: dict):
        """Process a single task"""
        task_id = task["id"]
        logger.info(f"Starting to process task {task_id}")
        try:
            # Store current task
            self.current_task = task
            
            # Get workflow path and prompt from task input
            workflow_path = self.get_workflow_path(task)
            prompt = task.get("input", {}).get("prompt")
            
            if not prompt:
                raise ValueError("No prompt provided in task input")
                
            logger.info(f"Using workflow: {workflow_path}")
            logger.info(f"Processing prompt: {prompt}")

            # Load and prepare workflow
            workflow = self.load_workflow(workflow_path)
            workflow = self.inject_prompt(workflow, prompt)
            logger.info("Workflow prepared with prompt")

            # Execute workflow and get results
            output_images = self.execute_workflow(workflow)
            
            if output_images:
                # Save and upload the first image
                output_path = f"outputs/{task_id}.png"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                output_images[0].save(output_path)
                
                # Upload image
                image_url = self.upload_file(output_path, task_id)
                logger.info(f"Image uploaded successfully: {image_url}")
                
                # Update task status
                self.update_task(
                    task_id,
                    status="success",
                    output={
                        "image_urls": [image_url]
                    }
                )
                
                # Cleanup
                os.remove(output_path)
                
            self.current_task = None
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            self.update_task(
                task_id,
                status="failed",
                error={
                    "code": 10001,
                    "message": str(e)
                }
            )
            self.current_task = None

    def claim_task(self) -> dict:
        """Try to claim a task from the API"""
        response = requests.post(
            f"{self.api_base}/tasks/claim",
            json={
                "worker_id": self.worker_id,
                "task_type": "text-to-image"
            },
            headers=self.headers
        )
        data = response.json()
        if data.get("id"):
            logger.info(f"Claimed task: {data}")
        return data

    def update_task(self, task_id: str, status: str, progress: int = None, **kwargs):
        """Update task status"""
        response = requests.post(
            f"{self.api_base}/tasks/{task_id}",
            json={
                "worker_id": self.worker_id,
                "status": status,
                "progress": progress,
                **kwargs
            },
            headers=self.headers
        )
        return response.json()

    def upload_file(self, file_path: str, task_id: str) -> str:
        """Upload a file to the asset management API"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'taskId': task_id}
            response = requests.post(
                f"{self.api_base}/tasks/{task_id}/assets",
                files=files,
                data=data,
                headers=self.headers
            )
            if response.status_code != 200:
                raise ValueError(f"Failed to upload file: {response.status_code}")
            return response.json()["url"]

    def load_workflow(self, workflow_path: str) -> dict:
        """Load the workflow JSON template"""
        with open(workflow_path, 'r') as f:
            return json.load(f)

    def inject_prompt(self, workflow: dict, prompt: str) -> dict:
        """Inject the prompt into the workflow"""
        # This is a simplified example - modify according to your workflow structure
        workflow["6"]["inputs"]["text"] = f'A 3D render of {prompt}, smooth lighting, no reflections, no shadows, keep the main subject center, 3d'
        return workflow

    def run(self):
        """Main worker loop"""
        logger.info(f"Worker {self.worker_id} started")

        while True:
            try:
                if not self.current_task:  # Only claim new task if no task is being processed
                    task = self.claim_task()
                    if task.get("id"):
                        logger.info(f"Processing task {task['id']}")
                        self.process_task(task)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}")
                time.sleep(5)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-base', help='Base URL for the API')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--comfy-url', help='ComfyUI server URL')
    args = parser.parse_args()
    
    # Create worker with explicit parameters, falling back to env vars
    worker = ComfyWorker(
        api_base=args.api_base,
        api_key=args.api_key,
        comfy_url=args.comfy_url
    )
    worker.run() 