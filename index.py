from tkinter import *
from flask import Flask,redirect, url_for,render_template,request
import os

def d_dtcn():
	root = Tk()
	root.configure(background = "black")

	def function1():
		os.system("python drowsiness_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat")
		exit()

	for i in range(5):
		root.grid_rowconfigure(i, weight=1)
	for j in range(5):
		root.grid_columnconfigure(j, weight=1)

	root.title("DROWSENSE MODE")
	bg_color = "#ff004f"
	button_color = "#ff004f"
	text_color = "white"
	font_style = "Helvetica 20"



	run_button = Button(root, text="Run using webcam", font=(font_style, 20), bg=button_color, fg=text_color,
						command=function1, relief="solid", borderwidth=1)
	run_button.grid(row=5, columnspan=5, sticky="wens", padx=5, pady=5)

	exit_button = Button(root, text="Exit", font=(font_style, 20), bg=button_color, fg=text_color, command=root.destroy,
						 relief="solid", borderwidth=1)
	exit_button.grid(row=9, columnspan=5, sticky="wens", padx=5, pady=5)

	root.mainloop()