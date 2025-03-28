podman run --detach \
           --name gemini-backend-container \
           --publish 8000:8000 \
           --volume "$(pwd)/.env:/app/.env:ro,Z" \
           --volume "$(pwd)/../projects:/app/../projects:ro,Z" \
           localhost/gemini-chat-backend:0.1
