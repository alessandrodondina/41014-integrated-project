import glfw
import time

print("Testing GLFW...", flush=True)

def error_callback(code, description):
    print(f"GLFW ERROR {code}: {description}", flush=True)

glfw.set_error_callback(error_callback)

if not glfw.init():
    raise RuntimeError("GLFW initialization failed")

window = glfw.create_window(640, 480, "41014 Graphics Test", None, None)

if not window:
    glfw.terminate()
    raise RuntimeError("GLFW window creation failed")

print("SUCCESS: Graphics window created!", flush=True)

glfw.make_context_current(window)

for _ in range(300):
    if glfw.window_should_close(window):
        break
    glfw.poll_events()
    glfw.swap_buffers(window)
    time.sleep(0.01)

glfw.destroy_window(window)
glfw.terminate()

print("Graphics test finished.", flush=True)
