import tkinter as tk
from tkinter import filedialog, Toplevel, Label, Button, Entry, Frame
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model
model_path = 'fine_tuned_spoof_model_600x600.h5'
spoof_model = load_model(model_path)

# Function to preprocess image for prediction
def preprocess_image(img_path, target_size=(600, 600)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict if the face is real or fake
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = spoof_model.predict(img_array)
    real_prob = prediction[0][0] * 100
    fake_prob = (1 - prediction[0][0]) * 100
    return real_prob, fake_prob

# Function to show the upload page after login
def show_upload_page():
    main_frame.pack_forget()
    login_popup.destroy()
    upload_frame.pack(fill='both', expand=True)

# Function to handle login
def open_login_popup():
    def login():
        email = email_entry.get()
        password = password_entry.get()
        if email and password:  # Basic check for non-empty fields
            show_upload_page()
        else:
            error_label.config(text="Please enter both email and password")

    global login_popup
    login_popup = Toplevel(root)
    login_popup.title("Login")
    login_popup.geometry("300x250")
    login_popup.configure(bg="#1c1c1c")

    Label(login_popup, text="Login", font=("Arial", 16, "bold"), bg="#1c1c1c", fg="#00ffcc").pack(pady=10)

    Label(login_popup, text="Email:", font=("Arial", 12), bg="#1c1c1c", fg="#f8f9fa").pack(anchor="w", padx=20)
    email_entry = Entry(login_popup, font=("Arial", 12), width=30)
    email_entry.pack(pady=5, padx=20)

    Label(login_popup, text="Password:", font=("Arial", 12), bg="#1c1c1c", fg="#f8f9fa").pack(anchor="w", padx=20)
    password_entry = Entry(login_popup, font=("Arial", 12), width=30, show="*")
    password_entry.pack(pady=5, padx=20)

    error_label = Label(login_popup, text="", font=("Arial", 10), bg="#1c1c1c", fg="red")
    error_label.pack(pady=5)

    submit_button = Button(login_popup, text="Submit", command=login, bg="#00ffcc", fg="#1c1c1c", relief="flat", font=("Arial", 12))
    submit_button.pack(pady=10)

# Function to show the bar chart in the specified area
def show_chart():
    # Clear any previous chart
    for widget in chart_frame.winfo_children():
        widget.destroy()

    # Create a bar graph
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = ['Real', 'Fake']
    real_prob, fake_prob = predict_image(file_path)
    values = [real_prob, fake_prob]
    ax.bar(labels, values, color=['#4caf50', '#f44336'])

    ax.set_ylim(0, 100)  # Set y-axis to range from 0 to 100
    ax.set_ylabel('Probability (%)')
    ax.set_title('Face Spoofing Detection Result')

    # Embed the bar chart in the Tkinter window (within the chart_frame)
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=20)

# Function to upload and process the image
def upload_image():
    global img_label, real_prob_label, fake_prob_label, file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        # Display the image
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        # Predict probabilities
        real_prob, fake_prob = predict_image(file_path)

        # Update result labels dynamically
        real_prob_label.config(text=f"Real Face Probability: {real_prob:.2f}%", fg="#4caf50")
        fake_prob_label.config(text=f"Fake Face Probability: {fake_prob:.2f}%", fg="#f44336")

# Create the Tkinter app
root = tk.Tk()
root.title("Face Spoofing Detection")
root.geometry("1200x800")
root.configure(bg="#1c1c1c")  # Darker theme background
root.state('zoomed')  # Make the window full-screen

# Load the background image
bg_image_path = r"C:\Users\Aadya\OneDrive\Desktop\final\Background.jpg"  # Replace with your image path
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Label widget to display the background image
bg_label = Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# First Page: Main Frame
main_frame = Frame(root, bg="#222222")  # Lighter background color for the box
main_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.6, relheight=0.4)

header = Label(main_frame, text="FACE SPOOFING DETECTION", font=("Helvetica", 36, "bold"), bg="#222222", fg="#ffffff")
header.pack(pady=30)

info_text = (
    "Face spoofing is an attempt to bypass a facial recognition system by using a photo or another substitute "
    "for the real face. This type of attack can compromise security systems that rely on facial recognition, leading to "
    "unauthorized access. Face spoofing can be performed using printed photos"
)

info_label = Label(main_frame, text=info_text, font=("Arial", 14), bg="#222222", fg="#ffffff", wraplength=800, justify="left")
info_label.pack(pady=10, padx=20)

login_button = Button(main_frame, text="Login", command=open_login_popup, font=("Arial", 14), bg="#00ffcc", fg="#1c1c1c", activebackground="#00e6b8", activeforeground="#1c1c1c", relief="flat", padx=20, pady=10)
login_button.pack(pady=20)

# Second Page: Upload Frame with split screen
upload_frame = Frame(root, bg="#1c1c1c")

# Left side: Explanation, Upload, and Result
left_frame = Frame(upload_frame, bg="#1c1c1c", width=600)
left_frame.pack(side="left", fill="both", expand=True)

header = Label(left_frame, text="FACE SPOOFING DETECTION", font=("Helvetica", 36, "bold"), bg="#1c1c1c", fg="#ffffff")
header.pack(pady=30)

# Short explanation on left side
explanation_text = (
    "Face spoofing is a technique used to bypass facial recognition systems by using fake images or videos."
)

explanation_label = Label(left_frame, text=explanation_text, font=("Arial", 12), bg="#1c1c1c", fg="#ffffff", wraplength=600, justify="left")
explanation_label.pack(pady=10, padx=20)

upload_button = Button(left_frame, text="Upload Image", command=upload_image, font=("Arial", 14), bg="#00ffcc", fg="#1c1c1c", activebackground="#00e6b8", activeforeground="#1c1c1c", relief="flat", padx=20, pady=10)
upload_button.pack(pady=20)

img_label = Label(left_frame, bg="#1c1c1c")
img_label.pack(pady=20)

real_prob_label = Label(left_frame, text="", font=("Arial", 16), bg="#1c1c1c", fg="#4caf50")
real_prob_label.pack(anchor="w", padx=20)

fake_prob_label = Label(left_frame, text="", font=("Arial", 16), bg="#1c1c1c", fg="#f44336")
fake_prob_label.pack(anchor="w", padx=20)

# Right side: View Chart Button and chart area
right_frame = Frame(upload_frame, bg="#1c1c1c", width=600)
right_frame.pack(side="right", fill="both", expand=True)

chart_button = Button(right_frame, text="View Chart", command=show_chart, font=("Arial", 14), bg="#00ffcc", fg="#1c1c1c", activebackground="#00e6b8", activeforeground="#1c1c1c", relief="flat", padx=20, pady=10)
chart_button.pack(pady=20)

# Chart frame (in the right side, within the right frame)
chart_frame = Frame(right_frame, bg="#1c1c1c", width=500, height=500)
chart_frame.pack(pady=20)

# Footer Section

# Run the app
root.mainloop()
