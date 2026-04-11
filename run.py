import os

mode = os.getenv("MODE", "api")

if mode == "api":
    # Phase 1 (OpenEnv)
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

else:
    # Phase 2 (Inference)
    import inference
    inference.main()