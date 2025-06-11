import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps
import time
import json
import modules.globals
import modules.metadata
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    resolve_relative_path,
)




from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager
import platform
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960 #instrad of 960
PREVIEW_DEFAULT_HEIGHT = 540 #instead of 540

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = (740,)
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = (890,)
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_OUTPUT = None

_ = None
preview_label = None
preview_slider = None
source_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
source_label_dict_live = {}

img_ft, vid_ft = modules.globals.file_types


def init(destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._
    ROOT = create_root(destroy)
    PREVIEW = create_preview(ROOT)
    return ROOT


def save_switch_states():
    switch_states = {
        "keep_audio": modules.globals.keep_audio,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get(
            "show_mouth_mask_box", False
        )
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass


def create_root(destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, status_label, show_fps_switch

    load_switch_states()

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    select_face_button = ctk.CTkButton(
        root, text=_("Select a face"), cursor="hand2", command=lambda: select_source_path()
    )
    select_face_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

 

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui["face_enhancer"])
    enhancer_switch = ctk.CTkSwitch(
        root,
        text=_("Face Enhancer"),
        variable=enhancer_value,
        cursor="hand2",
        command=lambda: (
            update_tumbler("face_enhancer", enhancer_value.get()),
            save_switch_states(),
        ),
    )
    enhancer_switch.place(relx=0.1, rely=0.6)

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(
        root,
        text=_("Keep audio"),
        variable=keep_audio_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_audio", keep_audio_value.get()),
            save_switch_states(),
        ),
    )
    keep_audio_switch.place(relx=0.1, rely=0.7)
   

    #    nsfw_value = ctk.BooleanVar(value=modules.globals.nsfw_filter)
    #    nsfw_switch = ctk.CTkSwitch(root, text='NSFW filter', variable=nsfw_value, cursor='hand2', command=lambda: setattr(modules.globals, 'nsfw_filter', nsfw_value.get()))
    #    nsfw_switch.place(relx=0.6, rely=0.7)


    show_fps_value = ctk.BooleanVar(value=modules.globals.show_fps)
    show_fps_switch = ctk.CTkSwitch(
        root,
        text=_("Show FPS"),
        variable=show_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "show_fps", show_fps_value.get()),
            save_switch_states(),
        ),
    )
    show_fps_switch.place(relx=0.1, rely=0.75)

    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    mouth_mask_switch = ctk.CTkSwitch(
        root,
        text=_("Mouth Mask"),
        variable=mouth_mask_var,
        cursor="hand2",
        command=lambda: setattr(modules.globals, "mouth_mask", mouth_mask_var.get()),
    )
    mouth_mask_switch.place(relx=0.1, rely=0.55)

    show_mouth_mask_box_var = ctk.BooleanVar(value=modules.globals.show_mouth_mask_box)
    show_mouth_mask_box_switch = ctk.CTkSwitch(
        root,
        text=_("Show Mouth Mask Box"),
        variable=show_mouth_mask_box_var,
        cursor="hand2",
        command=lambda: setattr(
            modules.globals, "show_mouth_mask_box", show_mouth_mask_box_var.get()
        ),
    )
    show_mouth_mask_box_switch.place(relx=0.1, rely=0.65)

    ctk.CTkButton(root, text=_("Destroy"), cursor="hand2", command=lambda: destroy())

 
    # --- Camera Selection ---
    camera_label = ctk.CTkLabel(root, text=_("Select Camera:"))
    camera_label.place(relx=0.1, rely=0.86, relwidth=0.2, relheight=0.05)

    available_cameras = get_available_cameras()
    camera_indices, camera_names = available_cameras

    if not camera_names or camera_names[0] == "No cameras found":
        camera_variable = ctk.StringVar(value="No cameras found")
        camera_optionmenu = ctk.CTkOptionMenu(
            root,
            variable=camera_variable,
            values=["No cameras found"],
            state="disabled",
        )
    else:
        camera_variable = ctk.StringVar(value=camera_names[0])
        camera_optionmenu = ctk.CTkOptionMenu(
            root, variable=camera_variable, values=camera_names
        )

    camera_optionmenu.place(relx=0.35, rely=0.86, relwidth=0.25, relheight=0.05)

    live_button = ctk.CTkButton(
        root,
        text=_("Live"),
        cursor="hand2",
        command=lambda: webcam_preview(
            root,
            (
                camera_indices[camera_names.index(camera_variable.get())]
                if camera_names and camera_names[0] != "No cameras found"
                else None
            ),
        ),
        state=(
            "normal"
            if camera_names and camera_names[0] != "No cameras found"
            else "disabled"
        ),
    )
    live_button.place(relx=0.65, rely=0.86, relwidth=0.2, relheight=0.05)
    # --- End Camera Selection ---

    status_label = ctk.CTkLabel(root, text=None, justify="center")
    status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(
        root, text="Deep Live Cam", justify="center", cursor="hand2"
    )
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open("https://deeplivecam.net")
    )

    return root



def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(_("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

  

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=_(text))
    ROOT.update()


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, update the frame processors
    if PREVIEW.state() == "normal":
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title=_("select a source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def fit_image_to_size(image, width: int, height: int):
    if width is None or height is None or width <= 0 or height <= 0:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    ratio_w = width / w
    ratio_h = height / h
    # Use the smaller ratio to ensure the image fits within the given dimensions
    ratio = min(ratio_w, ratio_h)
    
    # Compute new dimensions, ensuring they're at least 1 pixel
    new_width = max(1, int(ratio * w))
    new_height = max(1, int(ratio * h))
    new_size = (new_width, new_height)

    return cv2.resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def toggle_preview() -> None:

    PREVIEW.withdraw()



def webcam_preview(root: ctk.CTk, camera_index: int):
    global POPUP_LIVE

    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return

   
    if modules.globals.source_path is None:
        update_status("Please select a source image first")
        return
    create_webcam_preview(camera_index)


def get_available_cameras():
    """Returns a list of available camera names and indices."""
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()

            # Create list of indices and names
            camera_indices = list(range(len(devices)))
            camera_names = devices

            # If no cameras found through DirectShow, try OpenCV fallback
            if not camera_names:
                # Try to open camera with index -1 and 0
                test_indices = [-1, 0]
                working_cameras = []

                for idx in test_indices:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        working_cameras.append(f"Camera {idx}")
                        cap.release()

                if working_cameras:
                    return test_indices[: len(working_cameras)], working_cameras

            # If still no cameras found, return empty lists
            if not camera_names:
                return [], ["No cameras found"]

            return camera_indices, camera_names

        except Exception as e:
            print(f"Error detecting cameras: {str(e)}")
            return [], ["No cameras found"]
    else:
        # Unix-like systems (Linux/Mac) camera detection
        camera_indices = []
        camera_names = []

        if platform.system() == "Darwin":  # macOS specific handling
            # Try to open the default FaceTime camera first
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                camera_indices.append(0)
                camera_names.append("FaceTime Camera")
                cap.release()

            # On macOS, additional cameras typically use indices 1 and 2
            for i in [1, 2]:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()
        else:
            # Linux camera detection - test first 10 indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()

        if not camera_names:
            return [], ["No cameras found"]

        return camera_indices, camera_names





# @line_profiler.profile
def create_webcam_preview(camera_index: int):
    global preview_label, PREVIEW
    cap = VideoCapturer(camera_index)
    if not cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 120):
        update_status("Failed to start camera")
        return

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    PREVIEW.deiconify()

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()        
        if not ret:
            break
        temp_frame = frame.copy()
        if modules.globals.live_mirror:
            temp_frame = cv2.flip(temp_frame, 1)

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )
            frame = fit_image_to_size(frame, PREVIEW.winfo_width(), PREVIEW.winfo_height())  


        else:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )
            frame = fit_image_to_size(frame, PREVIEW.winfo_width(), PREVIEW.winfo_height())  

        for frame_processor in frame_processors:
            if frame_processor.NAME == "DLC.FACE-ENHANCER":
                if modules.globals.fp_ui["face_enhancer"]:
                    temp_frame, target_face = frame_processor.process_frame(None, temp_frame)
            else:
                temp_frame, target_face = frame_processor.process_frame(None, temp_frame)

        # Calculate and display FPS
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time


        if target_face:

            blur_weight = 5
            face_coords = target_face["bbox"]
            cur_preview_height, cur_preview_width = temp_frame.shape[:2]
              
            temp_frame[:round(face_coords[1]), :cur_preview_width] = frame[:round(face_coords[1]), :cur_preview_width]
            temp_frame[:cur_preview_height, :round(face_coords[0])] = frame[:cur_preview_height, :round(face_coords[0])]
            temp_frame[:cur_preview_height, round(face_coords[2]):cur_preview_width] = frame[:cur_preview_height, round(face_coords[2]):cur_preview_width]
            temp_frame[round(face_coords[3]):cur_preview_height, :cur_preview_width] = frame[round(face_coords[3]):cur_preview_height, :cur_preview_width]

            if round(face_coords[1]):
                top_shelf = temp_frame[:round(face_coords[1]), :cur_preview_width]
                blurred_top_shelf = cv2.GaussianBlur(top_shelf, (blur_weight, blur_weight), 0)
                temp_frame[:round(face_coords[1]), :cur_preview_width]  = blurred_top_shelf
            if round(face_coords[0]):
                left_shelf = temp_frame[:cur_preview_height, :round(face_coords[0])]
                blurred_left_shelf = cv2.GaussianBlur(left_shelf, (blur_weight, blur_weight), 0)
                temp_frame[:cur_preview_height, :round(face_coords[0])] = blurred_left_shelf

            if round(face_coords[2])<cur_preview_width:
                right_shelf = temp_frame[:cur_preview_height, round(face_coords[2]):cur_preview_width]
                blurred_right_shelf = cv2.GaussianBlur(right_shelf, (blur_weight, blur_weight), 0)
                temp_frame[:cur_preview_height, round(face_coords[2]):cur_preview_width] = blurred_right_shelf

            if round(face_coords[3])<cur_preview_height:
                bottom_shelf = temp_frame[round(face_coords[3]):cur_preview_height, :cur_preview_width]
                blurred_bottom_shelf = cv2.GaussianBlur(bottom_shelf, (blur_weight, blur_weight), 0)
                temp_frame[round(face_coords[3]):cur_preview_height, :cur_preview_width] = blurred_bottom_shelf
                
        if modules.globals.show_fps:
            cv2.putText(
                temp_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageOps.contain(
            image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        ROOT.update()

        if PREVIEW.state() == "withdrawn":
            break

    cap.release()
    PREVIEW.withdraw()

