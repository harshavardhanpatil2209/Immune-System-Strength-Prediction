import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt

# Load the datasets (replace file paths with your actual file paths)
platelet_file_path = 'your_dataset.csv'
wbc_file_path = 'wbc_dataset.csv'
immuno_file_path = 'immuno_dataset.csv'
tcell_file_path = 'tcell_dataset.csv'
bcell_file_path = 'bcell_dataset.csv'

# Load datasets into DataFrames
df_platelets = pd.read_csv(platelet_file_path)
df_wbc = pd.read_csv(wbc_file_path)
df_immuno = pd.read_csv(immuno_file_path)
df_tcell = pd.read_csv(tcell_file_path)
df_bcell = pd.read_csv(bcell_file_path)

# Function to train linear regression model and scale features
def train_model_and_scale(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    return scaler, model

# Train models and scale features for each dataset
scaler_platelets, model_platelets = train_model_and_scale(df_platelets[['Platelets_Count']], df_platelets['Percentage_Value'])
scaler_wbc, model_wbc = train_model_and_scale(df_wbc[['WBC_Count']], df_wbc['Percentage_Value'])
scaler_immuno, model_immuno = train_model_and_scale(df_immuno[['Immunoglobulin_Count']], df_immuno['Percentage_Value'])
scaler_tcell, model_tcell = train_model_and_scale(df_tcell[['tcell_Count']], df_tcell['Percentage_Value'])
scaler_bcell, model_bcell = train_model_and_scale(df_bcell[['bcell_Count']], df_bcell['Percentage_Value'])

# Function to predict percentage value based on user input
def predict_percentage(user_input, scaler, model):
    user_input_scaled = scaler.transform(np.array(user_input).reshape(-1, 1))
    percentage_value = model.predict(user_input_scaled)[0]
    return percentage_value

# Function to handle prediction button click
def predict():
    try:
        platelets_count = float(platelets_entry.get())
        wbc_count = float(wbc_entry.get())
        immuno_count = float(immuno_entry.get())
        tcell_count = float(tcell_entry.get())
        bcell_count = float(bcell_entry.get())

        platelets_percentage = predict_percentage(platelets_count, scaler_platelets, model_platelets)
        wbc_percentage = predict_percentage(wbc_count, scaler_wbc, model_wbc)
        immuno_percentage = predict_percentage(immuno_count, scaler_immuno, model_immuno)
        tcell_percentage = predict_percentage(tcell_count, scaler_tcell, model_tcell)
        bcell_percentage = predict_percentage(bcell_count, scaler_bcell, model_bcell)

        # Calculate average percentage
        avg_percentage = (platelets_percentage + wbc_percentage + immuno_percentage + tcell_percentage + bcell_percentage) / 5

        # Determine health status
        health_status = "Healthy" if avg_percentage > 33 else "Unhealthy"

        # Display predicted percentages and health status
        messagebox.showinfo("Predictions and Health Status",
                            f"Predicted Percentage Value for Platelets : {platelets_percentage:.2f}\n"
                            f"Predicted Percentage Value for WBC : {wbc_percentage:.2f}\n"
                            f"Predicted Percentage Value for Immunoglobins : {immuno_percentage:.2f}\n"
                            f"Predicted Percentage Value for T cell : {tcell_percentage:.2f}\n"
                            f"Predicted Percentage Value for B cell: {bcell_percentage:.2f}\n"
                            f"Average Percentage: {avg_percentage:.2f}\n"
                            f"Health Status: {health_status}")

        # Plotting for all parameters in a single figure
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()

        for ax in axs:
            ax.grid(True)

        # Plot for Platelets
        axs[0].scatter(df_platelets['Platelets_Count'], df_platelets['Percentage_Value'], color='black', label='True values')
        axs[0].plot(df_platelets['Platelets_Count'], model_platelets.predict(scaler_platelets.transform(df_platelets[['Platelets_Count']])), color='blue', linewidth=3, label='Linear Regression Model')
        axs[0].scatter(platelets_count, platelets_percentage, color='red', label='User Input', marker='x', s=100)  # Add user input marker
        axs[0].set_xlabel('Platelets Count')
        axs[0].set_ylabel('Percentage Value')
        axs[0].set_title('Platelets')

        # Plot for WBC
        axs[1].scatter(df_wbc['WBC_Count'], df_wbc['Percentage_Value'], color='black', label='True values')
        axs[1].plot(df_wbc['WBC_Count'], model_wbc.predict(scaler_wbc.transform(df_wbc[['WBC_Count']])), color='blue', linewidth=3, label='Linear Regression Model')
        axs[1].scatter(wbc_count, wbc_percentage, color='red', label='User Input', marker='x', s=100)  # Add user input marker
        axs[1].set_xlabel('WBC Count')
        axs[1].set_ylabel('Percentage Value')
        axs[1].set_title('WBC')

        # Plot for Immunoglobins
        axs[2].scatter(df_immuno['Immunoglobulin_Count'], df_immuno['Percentage_Value'], color='black', label='True values')
        axs[2].plot(df_immuno['Immunoglobulin_Count'], model_immuno.predict(scaler_immuno.transform(df_immuno[['Immunoglobulin_Count']])), color='blue', linewidth=3, label='Linear Regression Model')
        axs[2].scatter(immuno_count, immuno_percentage, color='red', label='User Input', marker='x', s=100)  # Add user input marker
        axs[2].set_xlabel('Immunoglobins Count')
        axs[2].set_ylabel('Percentage Value')
        axs[2].set_title('Immunoglobins')

        # Plot for T cell
        axs[3].scatter(df_tcell['tcell_Count'], df_tcell['Percentage_Value'], color='black', label='True values')
        axs[3].plot(df_tcell['tcell_Count'], model_tcell.predict(scaler_tcell.transform(df_tcell[['tcell_Count']])), color='blue', linewidth=3, label='Linear Regression Model')
        axs[3].scatter(tcell_count, tcell_percentage, color='red', label='User Input', marker='x', s=100)  # Add user input marker
        axs[3].set_xlabel('T cell Count')
        axs[3].set_ylabel('Percentage Value')
        axs[3].set_title('T cell')

        # Plot for B cell
        axs[4].scatter(df_bcell['bcell_Count'], df_bcell['Percentage_Value'], color='black', label='True values')
        axs[4].plot(df_bcell['bcell_Count'], model_bcell.predict(scaler_bcell.transform(df_bcell[['bcell_Count']])), color='blue', linewidth=3, label='Linear Regression Model')
        axs[4].scatter(bcell_count, bcell_percentage, color='red', label='User Input', marker='x', s=100)  # Add user input marker
        axs[4].set_xlabel('B cell Count')
        axs[4].set_ylabel('Percentage Value')
        axs[4].set_title('B cell')

        # Remove the empty subplot
        fig.delaxes(axs[5])

        plt.tight_layout()
        plt.show()



        # Ask user if they want to make another prediction
        if messagebox.askyesno("Continue?", "Do you want to make another prediction?"):
            # Clear entry fields for next prediction
            platelets_entry.delete(0, tk.END)
            wbc_entry.delete(0, tk.END)
            immuno_entry.delete(0, tk.END)
            tcell_entry.delete(0, tk.END)
            bcell_entry.delete(0, tk.END)
        else:
            # Exit the application
            root.destroy()

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# Create the main application window
root = tk.Tk()
root.title("Medical Data Analysis")

# Create and configure the main frame
main_frame = ttk.Frame(root, padding="20")
main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Add labels and entry fields for each parameter
platelets_label = ttk.Label(main_frame, text="Platelets Count (150000 to 450000):")
platelets_label.grid(column=0, row=0, sticky=tk.W)
platelets_entry = ttk.Entry(main_frame)
platelets_entry.grid(column=1, row=0)

wbc_label = ttk.Label(main_frame, text="WBC Count (4500 to 11000):")
wbc_label.grid(column=0, row=1, sticky=tk.W)
wbc_entry = ttk.Entry(main_frame)
wbc_entry.grid(column=1, row=1)

immuno_label = ttk.Label(main_frame, text="Immunoglobins Count (70 to 400):")
immuno_label.grid(column=0, row=2, sticky=tk.W)
immuno_entry = ttk.Entry(main_frame)
immuno_entry.grid(column=1, row=2)

tcell_label = ttk.Label(main_frame, text="T cell Count (500 to 1200):")
tcell_label.grid(column=0, row=3, sticky=tk.W)
tcell_entry = ttk.Entry(main_frame)
tcell_entry.grid(column=1, row=3)

bcell_label = ttk.Label(main_frame, text="B cell Count (1000 to 4800):")
bcell_label.grid(column=0, row=4, sticky=tk.W)
bcell_entry = ttk.Entry(main_frame)
bcell_entry.grid(column=1, row=4)

# Add a button to trigger predictions
predict_button = ttk.Button(main_frame, text="Predict", command=predict)
predict_button.grid(column=0, row=5, columnspan=2, pady=10)

# Start the GUI event loop
root.mainloop()
