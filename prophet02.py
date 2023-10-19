import pandas as pd
import threading
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import os
from numpy import hstack

def generate_output_filename():
    weights_str = "-".join([f"{key}={value.get()}" for key, value in weights_labels])
    base_filename = f"output-{weights_str}-DBSCAN.csv"
    version = 1
    
    while True:
        potential_path = os.path.join(output_dir_var.get(), base_filename)
        if os.path.exists(potential_path):
            base_filename = f"output-{weights_str}-DBSCAN-v{version}.csv"
            version += 1
        else:
            break
    
    return os.path.join(output_dir_var.get(), base_filename)

def start_process_data():
    process_btn.state(['disabled'])
    threading.Thread(target=process_data).start()

def process_data():
    try:
        customer_file = customer_input_var.get()
        non_customer_file = non_customer_input_var.get()

        if not (customer_file and non_customer_file):
            raise ValueError("Please select all files!")

        customers = pd.read_csv(customer_file, delimiter=";", encoding="ISO-8859-1")
        non_customers = pd.read_csv(non_customer_file, delimiter=";", encoding="ISO-8859-1")

        try:
            weights = {
                "Accountname": float(accountname_weight_var.get()),
                "Branche": float(branche_weight_var.get()),
                "Mitarbeiter": float(mitarbeiter_weight_var.get()),
                "Bundesland (Rechnungsanschrift)": float(bundesland_weight_var.get()),
                "Potential Unternehmen": float(potential_weight_var.get()),
                "Non-STST-Postings 3-MR": float(postings_weight_var.get()),
                "Website": float(website_weight_var.get())
            }
            eps_value = float(eps_var.get())
            min_samples_value = int(min_samples_var.get())
        except ValueError:
            raise ValueError("Please enter valid weighting values and DBSCAN parameters!")

        columns = list(weights.keys())
        X_customers_list = []
        X_non_customers_list = []

        for idx, col in enumerate(columns):
            vectorizer = TfidfVectorizer()
            customer_transformed = vectorizer.fit_transform(customers[col].astype(str)) * (weights[col] / 100.0)
            non_customer_transformed = vectorizer.transform(non_customers[col].astype(str)) * (weights[col] / 100.0)
    
            X_customers_list.append(customer_transformed)
            X_non_customers_list.append(non_customer_transformed)
    
            progress_var.set((idx + 1) * 5)
            progress_label_var.set(f"Transforming Data Column: {col}")
            root.update_idletasks()

        X_customers = hstack(X_customers_list)
        X_non_customers = hstack(X_non_customers_list)

        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        customer_labels = dbscan.fit_predict(X_customers)

        # Note: label = -1 means the point is an outlier in DBSCAN
        unique_labels = set(customer_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        progress_label_var.set("Predicting Potential Customers")
        non_customer_series = pd.Series(dbscan.predict(X_non_customers))
        potential_customers = non_customers[non_customer_series.isin(unique_labels)]
        output_file = generate_output_filename()
        potential_customers.to_csv(output_file, sep=';', encoding="latin1", index=False)   
        
        progress_var.set(100)
        progress_label_var.set("Process Completed.")
        messagebox.showinfo("Process Completed", "Data processing successfully completed!")
        process_btn.state(['!disabled'])

    except FileNotFoundError:
        messagebox.showerror("Error", "The file could not be found.")
        process_btn.state(['!disabled'])
    except Exception as e:
        messagebox.showerror("Error", str(e))
        process_btn.state(['!disabled'])

root = Tk()
root.title("profitprophet")

output_dir_var = StringVar()

frame = ttk.Frame(root, padding="30")
frame.grid(row=0, column=0, sticky=(W, E, N, S))

customer_input_var = StringVar()
non_customer_input_var = StringVar()
progress_var = IntVar()
progress_label_var = StringVar()

accountname_weight_var = StringVar(value="100")
branche_weight_var = StringVar(value="100")
mitarbeiter_weight_var = StringVar(value="100")
bundesland_weight_var = StringVar(value="100")
potential_weight_var = StringVar(value="100")
postings_weight_var = StringVar(value="100")
website_weight_var = StringVar(value="100")

eps_var = StringVar(value="0.5")
min_samples_var = StringVar(value="5")

ttk.Label(frame, text="Select Customer File:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
ttk.Entry(frame, textvariable=customer_input_var).grid(row=0, column=1, padx=5, pady=5)
ttk.Button(frame, text="Browse", command=lambda: customer_input_var.set(filedialog.askopenfilename())).grid(row=0, column=2, padx=5, pady=5)

ttk.Label(frame, text="Select Non-Customer File:").grid(row=1, column=0, padx=5, pady=5, sticky=W)
ttk.Entry(frame, textvariable=non_customer_input_var).grid(row=1, column=1, padx=5, pady=5)
ttk.Button(frame, text="Browse", command=lambda: non_customer_input_var.set(filedialog.askopenfilename())).grid(row=1, column=2, padx=5, pady=5)

ttk.Label(frame, text="Select Output Directory:").grid(row=2, column=0, padx=5, pady=5, sticky=W)
ttk.Entry(frame, textvariable=output_dir_var).grid(row=2, column=1, padx=5, pady=5)
ttk.Button(frame, text="Browse", command=lambda: output_dir_var.set(filedialog.askdirectory())).grid(row=2, column=2, padx=5, pady=5)

weights_labels = [
    ("Accountname", accountname_weight_var),
    ("Branche", branche_weight_var),
    ("Mitarbeiter", mitarbeiter_weight_var),
    ("Bundesland (Rechnungsanschrift)", bundesland_weight_var),
    ("Potential Unternehmen", potential_weight_var),
    ("Non-STST-Postings 3-MR", postings_weight_var),
    ("Website", website_weight_var)
]

# Here are your UI elements...

for idx, (text, var) in enumerate(weights_labels):
    ttk.Label(frame, text=f"{text} Weight (%)").grid(row=idx+3, column=0, padx=5, pady=5, sticky=W)
    ttk.Entry(frame, textvariable=var, width=5).grid(row=idx+3, column=1, padx=5, pady=5)

ttk.Label(frame, text="EPS for DBSCAN").grid(row=10, column=0, padx=5, pady=5, sticky=W)
ttk.Entry(frame, textvariable=eps_var, width=5).grid(row=10, column=1, padx=5, pady=5)

ttk.Label(frame, text="Min samples for DBSCAN").grid(row=11, column=0, padx=5, pady=5, sticky=W)
ttk.Entry(frame, textvariable=min_samples_var, width=5).grid(row=11, column=1, padx=5, pady=5)

process_btn = ttk.Button(frame, text="Process Data", command=start_process_data)
process_btn.grid(row=12, column=0, columnspan=2, pady=20)

ttk.Progressbar(frame, variable=progress_var, maximum=100).grid(row=13, column=0, columnspan=2, sticky=(W, E), padx=5)
ttk.Label(frame, textvariable=progress_label_var).grid(row=14, column=0, columnspan=2, padx=5)

root.mainloop()