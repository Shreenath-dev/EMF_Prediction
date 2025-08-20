import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import numpy as np
import pandas as pd
from PIL import ImageTk,Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pyttsx3
from tkinter import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from tkinter import Tk, Label, Entry, Button, StringVar
df = pd.read_csv('C:\\Users\\Srinath\\OneDrive\\Others\\Desktop\\bit 2\\dataset.csv')
X = df[['Speed', 'Field Current']].values
y = df['Eg(output)'].values

model = LinearRegression()
model.fit(X, y)

def predict_eg():
    try:
        speed_input = float(field1.get())
        field_current_input = float(field2.get())
        new_data = np.array([[speed_input, field_current_input]])
        prediction = model.predict(new_data)
        result_var.set(f"Predicted Eg: {prediction[0]:.2f}")
    except ValueError:
        result_var.set("Invalid input. Please enter numeric values for Speed and Field Current.")

# ui 
root = Tk()
root.title("Eg Prediction")
bg = ImageTk.PhotoImage(file="C:\\Users\\Srinath\\OneDrive\\Others\Desktop\\bit 2\\bgbeni.png")
canvas = Canvas(root, width=350, height=197)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg, anchor="nw")
label1 = Label(root, text="Speed:")
label1.place(x=35,y=45)

field1 = Entry(root)
field1.place(x=200,y=45)

lbl2= Label(root, text="Field Current:")
lbl2.place(x=35,y=70)

field2 = Entry(root)
field2.place(x=200,y=70)

button1 = Button(root, text="Predict Eg", command=predict_eg)
button1.place(x=200,y=150)

result_var = StringVar()
result_label = Label(root, textvariable=result_var)
result_label.place(x=60, y=100)
#calling ui
root.mainloop()


