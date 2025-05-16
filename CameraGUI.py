import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import datetime

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Viewer")
        self.root.geometry("800x600")
        self.photo_count = 0

        # Layout dinamico
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        # Titolo
        self.title_label = tk.Label(root, bg="white")
        self.title_label.grid(row=0, column=0, pady=(10, 0))
        self.title_text = "Gender Recognition via Hands"

        # Descrizione
        self.description_label = tk.Label(root, bg="white")
        self.description_label.grid(row=1, column=0, pady=(5, 10))
        self.description_text = "Take a photo of your hand palm"

        # Area video
        self.video_frame = ttk.Frame(root)
        self.video_frame.grid(row=3, column=0, sticky="nsew")
        self.video_frame.columnconfigure(0, weight=1)
        self.video_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Bottoni
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

        # Eventi
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.bind("<Configure>", self.on_resize)

        # Disegna testo iniziale
        self.update_texts()

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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            label = "palm" if self.photo_count == 0 else "dorsal"
            filename = f"hand_{label}_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            print(f"Foto salvata: {filename}")

            self.photo_count += 1
            if self.photo_count == 1:
                self.description_text = "Take a photo of your hand dorsal"
                self.update_texts()

    def quit_app(self):
        self.cap.release()
        self.root.destroy()

    def update_antialiased_text(self, text, widget, base_size=20, height=40, scale_ref=800):
        width = max(self.root.winfo_width(), 300)  # Usa minimo per evitare 0 o 1
        scaled_size = max(10, int(base_size * width / scale_ref))  # Font scalabile

        img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", scaled_size)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = ((width - text_width) // 2, (height - text_height) // 2)

        draw.text(text_position, text, font=font, fill="black")

        photo = ImageTk.PhotoImage(img)
        widget.configure(image=photo)
        widget.image = photo

    def update_texts(self):
        self.update_antialiased_text(self.title_text, self.title_label, base_size=24, height=50)
        self.update_antialiased_text(self.description_text, self.description_label, base_size=16, height=35)

    def on_resize(self, event):
        self.root.after_idle(self.update_texts)  # Posticipa ridisegno per ottenere larghezza aggiornata

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
