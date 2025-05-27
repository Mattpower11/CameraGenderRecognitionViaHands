import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import datetime

from palm_cut import get_POI_hand

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Viewer")
        self.root.geometry("800x600")
        self.root.minsize(400, 300)
        self.photo_count = 0

        # Layout configurazione
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        # Titolo
        self.title_label = ttk.Label(root, text="Gender Recognition via Hands", font=("Helvetica", 20, "bold"))
        self.title_label.grid(row=0, column=0, pady=(10, 0))

        # Descrizione dinamica
        self.description_text = tk.StringVar()
        self.description_text.set("Take a photo of your hand palm")
        self.description_label = ttk.Label(root, textvariable=self.description_text, font=("Helvetica", 14))
        self.description_label.grid(row=1, column=0, pady=(5, 10))

        # Frame video
        self.video_frame = ttk.Frame(root)
        self.video_frame.grid(row=3, column=0, sticky="nsew")
        self.video_frame.columnconfigure(0, weight=1)
        self.video_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Frame bottoni
        self.button_frame = ttk.Frame(root)
        self.button_frame.grid(row=4, column=0, sticky="ew", pady=10)
        self.button_frame.columnconfigure((0, 1), weight=1)

        self.capture_button = ttk.Button(self.button_frame, text="Scatta Foto", command=self.capture_image)
        self.capture_button.grid(row=0, column=0, sticky="ew", padx=5)

        self.quit_button = ttk.Button(self.button_frame, text="Esci", command=self.quit_app)
        self.quit_button.grid(row=0, column=1, sticky="ew", padx=5)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.update_video()

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 0 and canvas_height > 0:
                image = image.resize((canvas_width, canvas_height))

            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(10, self.update_video)

    def capture_image(self):
        if hasattr(self, 'current_frame'):
            if self.photo_count >= 2:
                return  # Non permettere più di 2 foto

            if len(get_POI_hand(self.current_frame)) > 0:
                #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                label = "palmar" if self.photo_count == 0 else "dorsal"
                #filename = f"./photos/hand_{label}_{timestamp}.jpg"
                filename = f".\photos\hand_{label}.jpg"
                cv2.imwrite(filename, self.current_frame)
                print(f"Foto salvata: {filename}")

                self.photo_count += 1

                if self.photo_count == 1:
                    self.description_text.set("Take a photo of your hand dorsal")
                elif self.photo_count == 2:
                    self.description_text.set("Photo session completed. You can now exit.")
                    self.capture_button.config(state="disabled")
            else:
                print("Nessuna mano rilevata nell'immagine corrente.")
                self.previous_description = self.description_text.get()
                self.description_text.set("Nessuna mano rilevata. Riprova.")
                self.root.after(5000, self.reset_description)

    def reset_description(self):
        if hasattr(self, 'previous_description') and self.photo_count < 2:
            self.description_text.set(self.previous_description)
            del self.previous_description

    def quit_app(self):
        self.cap.release()
        self.root.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = WebcamApp(root)
#     root.mainloop()
