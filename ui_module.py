# ui_module.py
# GUI for user interaction, displaying statistics, model metrics, and plots using tkinter

# Required libraries for ui module
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import os
import threading
from load_dataset_module import DatasetLoader
from eda_module import DataAnalyzer

class UserInterface:
    def __init__(self, loader: DatasetLoader, analyzer: DataAnalyzer):
        self._loader = loader
        self._analyzer = analyzer

        # Initialize root window
        self._root = tk.Tk()
        self._root.title("Stroke Analysis Application")
        self._root.geometry("1000x800")
        self._root.minsize(700,500)

        # Main container frame
        self._main_frame = ttk.Frame(self._root, padding=10)
        self._main_frame.pack(fill=tk.BOTH, expand=True)

        # Buttons for workflow steps
        self._button_frame = ttk.Frame(self._main_frame)
        self._button_frame.grid(row=0, column=0, sticky="ew", pady=(0,10))

        self._button_widgets = {}
        buttons = [
            ("Load and Preprocess", self._load_data),
            ("Perform EDA", self._perform_eda),
            ("EDA Plots", lambda: self._display_plots('eda_plots', self._eda_plot_frame, 'EDA Plots')),
            ("Compute Features", self._compute_features),
            ("Train Models", self._start_train_models),
            ("ML Plots", lambda: self._display_plots('ml_plots', self._ml_plot_frame, 'ML Plots')),
            ("Exit", self._root.quit)
        ]
        for col, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(self._button_frame, text=text, command=cmd)
            btn.grid(row=0, column=col, padx=6, pady=2, sticky="ew")
            self._button_widgets[text] = btn

        # Make buttons expand evenly horizontally
        for col in range(len(buttons)):
            self._button_frame.grid_columnconfigure(col, weight=1)

        # Scrollable text output window for logs and results
        self._output_text = scrolledtext.ScrolledText(self._main_frame, height=14, wrap=tk.WORD, font=("Consolas", 12))
        self._output_text.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        # Tabbed interface for plots
        self._notebook = ttk.Notebook(self._main_frame)
        self._notebook.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        self._notebook_visible = False

        # EDA Plots tab
        self._eda_plot_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._eda_plot_frame, text="EDA Plots")
        self._eda_plot_canvas = tk.Canvas(self._eda_plot_frame, bg="#f9f9f9")
        self._eda_plot_scrollbar = ttk.Scrollbar(self._eda_plot_frame, orient="vertical", command=self._eda_plot_canvas.yview)
        self._eda_plot_canvas.configure(yscrollcommand=self._eda_plot_scrollbar.set)
        self._eda_inner_frame = ttk.Frame(self._eda_plot_canvas)
        self._eda_plot_canvas.create_window((0, 0), window=self._eda_inner_frame, anchor='nw')
        self._eda_inner_frame.bind("<Configure>", lambda e: self._eda_plot_canvas.configure(scrollregion=self._eda_plot_canvas.bbox("all")))
        self._eda_plot_canvas.grid(row=0, column=0, sticky="nsew")
        self._eda_plot_scrollbar.grid(row=0, column=1, sticky="ns")
        self._eda_plot_frame.grid_columnconfigure(0, weight=1)
        self._eda_plot_frame.grid_rowconfigure(0, weight=1)

        # ML Plots tab
        self._ml_plot_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._ml_plot_frame, text="ML Plots")
        self._ml_plot_canvas = tk.Canvas(self._ml_plot_frame, bg="#f9f9f9")
        self._ml_plot_scrollbar = ttk.Scrollbar(self._ml_plot_frame, orient="vertical", command=self._ml_plot_canvas.yview)
        self._ml_plot_canvas.configure(yscrollcommand=self._ml_plot_scrollbar.set)
        self._ml_inner_frame = ttk.Frame(self._ml_plot_canvas)
        self._ml_plot_canvas.create_window((0, 0), window=self._ml_inner_frame, anchor='nw')
        self._ml_inner_frame.bind("<Configure>", lambda e: self._ml_plot_canvas.configure(scrollregion=self._ml_plot_canvas.bbox("all")))
        self._ml_plot_canvas.grid(row=0, column=0, sticky="nsew")
        self._ml_plot_scrollbar.grid(row=0, column=1, sticky="ns")
        self._ml_plot_frame.grid_columnconfigure(0, weight=1)
        self._ml_plot_frame.grid_rowconfigure(0, weight=1)

        self._training_thread = None
        self._training_results = None

        # Configure grid weights for resizing behavior
        self._main_frame.grid_rowconfigure(1, weight=1)  # output text expands vertically
        self._main_frame.grid_rowconfigure(3, weight=2)  # notebook expands more vertically
        self._main_frame.grid_columnconfigure(0, weight=1)  # expand horizontally

    def _show_notebook(self):
        """Show the notebook if not visible."""
        if not self._notebook_visible:
            self._notebook.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
            self._notebook_visible = True

    def _hide_notebook(self):
        """Hide the notebook to save space."""
        if self._notebook_visible:
            self._notebook.grid_remove()
            self._notebook_visible = False

    def _load_data(self):
        """Load and preprocess the dataset."""
        try:
            self._hide_notebook()
            self._output_text.delete(1.0, tk.END)
            self._output_text.insert(tk.END, "Loading and preprocessing dataset...\n")
            self._loader.load_data()
            self._loader.preprocess_data()
            self._loader.split_data()
            self._output_text.insert(tk.END, " Dataset loaded, preprocessed, and split successfully!\n")
        except Exception as e:
            messagebox.showerror("Load Data Error", f"Failed to load dataset:\n{str(e)}")

    def _perform_eda(self):
        """Perform exploratory data analysis and show summary in output."""
        if self._loader._data is None:
            messagebox.showwarning("Warning", "Please load the dataset first before performing EDA.")
            return
        try:
            self._hide_notebook()
            self._output_text.delete(1.0, tk.END)
            self._output_text.insert(tk.END, "Performing Exploratory Data Analysis...\n")
            stats = self._analyzer.compute_statistics()
            self._output_text.insert(tk.END, "Descriptive Statistics:\n" + stats.to_string() + "\n\n")
            self._analyzer.visualize_data()
            self._analyzer.check_class_balance()
            self._output_text.insert(tk.END, " EDA completed. Visualizations saved to 'eda_plots' folder.\n")
        except Exception as e:
            messagebox.showerror("EDA Error", f"An error occurred during EDA:\n{str(e)}")

    def _compute_features(self):
        """Compute feature importances and correlations."""
        if self._loader._data is None:
            messagebox.showwarning("Warning", "Please load the dataset first before computing features.")
            return
        try:
            self._hide_notebook()
            self._output_text.delete(1.0, tk.END)
            self._output_text.insert(tk.END, "Computing feature importance and correlation...\n")
            features = self._analyzer.compute_statistical_features()
            self._output_text.insert(tk.END, " Feature importance saved in 'eda_plots/feature_importance.csv'.\n\n")
            self._output_text.insert(tk.END, "Feature correlation with Stroke Occurrence:\n")
            corr = features.get('Stroke Occurrence', {})
            for feat, val in corr.items():
                self._output_text.insert(tk.END, f"  {feat}: {val:.4f}\n")
        except Exception as e:
            messagebox.showerror("Feature Computation Error", f"Error computing features:\n{str(e)}")

    def _start_train_models(self):
        """Start model training on a background thread."""
        if self._loader._train_data is None:
            messagebox.showwarning("Warning", "Please load and preprocess the dataset first before training models.")
            return

        if self._training_thread and self._training_thread.is_alive():
            messagebox.showwarning("Warning", "Model training is already in progress.")
            return

        self._hide_notebook()
        self._output_text.delete(1.0, tk.END)
        self._output_text.insert(tk.END, "Training models... This may take a few minutes.\n")
        self._button_widgets["Train Models"].config(state="disabled")

        # Progress bar for long operations
        self._progress = ttk.Progressbar(self._main_frame, mode='indeterminate')
        self._progress.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._progress.start()

        # Run training in separate thread to keep GUI responsive
        self._training_thread = threading.Thread(target=self._train_models)
        self._training_thread.start()

        # Poll for training completion
        self._check_training_status()

    def _train_models(self):
        """Invoke analyzer to train and evaluate all models."""
        try:
            self._training_results = self._analyzer.train_and_evaluate_all()
        except Exception as e:
            self._training_results = str(e)

    def _check_training_status(self):
        """Check if training thread has finished and update GUI accordingly."""
        if self._training_thread.is_alive():
            self._root.after(100, self._check_training_status)
        else:
            self._button_widgets["Train Models"].config(state="normal")
            self._progress.stop()
            self._progress.grid_remove()

            if isinstance(self._training_results, str):
                # Training error
                messagebox.showerror("Training Error", self._training_results)
                self._output_text.insert(tk.END, f" Training failed:\n{self._training_results}\n")
            else:
                # Display training results
                self._output_text.insert(tk.END, " Training completed. Results:\n")
                for target, metrics in self._training_results.items():
                    self._output_text.insert(tk.END, f"\nTarget: {target}\n")
                    for model, scores in metrics.items():
                        self._output_text.insert(tk.END,
                            f"{model}:\n"
                            f"  Accuracy: {scores['accuracy']:.2f}\n"
                            f"  Precision: {scores['precision']:.2f}\n"
                            f"  Recall: {scores['recall']:.2f}\n"
                            f"  Confusion Matrix:\n{scores['confusion_matrix']}\n")
                self._output_text.insert(tk.END, "\nML plots saved in 'ml_plots' folder.\n")

    def _display_plots(self, plot_dir: str, plot_frame: ttk.Frame, tab_name: str):
        """Display all PNG plots from a directory in the specified frame."""
        if not os.path.exists(plot_dir):
            messagebox.showerror("Error", f"Plot directory '{plot_dir}' not found.")
            return

        self._show_notebook()
        self._notebook.select(plot_frame)  # Switch to the specified tab

        # Determine correct canvas and inner frame based on plot_dir
        if plot_dir == 'eda_plots':
            canvas = self._eda_plot_canvas
            inner_frame = self._eda_inner_frame
        elif plot_dir == 'ml_plots':
            canvas = self._ml_plot_canvas
            inner_frame = self._ml_inner_frame
        else:
            messagebox.showerror("Error", f"Unknown plot directory '{plot_dir}'")
            return
        
        # Clear previous images in the inner_frame only
        for widget in inner_frame.winfo_children():
            widget.destroy()


        plot_files = sorted([f for f in os.listdir(plot_dir) if f.lower().endswith('.png')])
        if not plot_files:
            self._output_text.insert(tk.END, f"No PNG plot images found in '{plot_dir}'.\n")
            return

        for idx, file in enumerate(plot_files):
            img_path = os.path.join(plot_dir, file)
            try:
                img = Image.open(img_path)
                # Resize image dynamically based on window width
                max_width = self._root.winfo_width() // 3 - 40  # Approx 1/3 of window width
                img.thumbnail((max_width, max_width * 2 // 3), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                # Place title above the plot
                title = ttk.Label(inner_frame, text=file.replace('_', ' ').replace('.png', '').capitalize())
                title.grid(row=(idx // 3) * 2, column=idx % 3, padx=10, pady=2)

                # Plot image below the title
                label = ttk.Label(inner_frame, image=photo, borderwidth=2, relief="groove")
                label.image = photo  # Keep reference
                label.grid(row=(idx // 3) * 2 + 1, column=idx % 3, padx=10, pady=10)
            except Exception as e:
                print(f"Error displaying image '{file}': {e}")

        self._output_text.insert(tk.END, f"Displayed {len(plot_files)} {tab_name} in '{tab_name}' tab.\n")

    def run(self):
        """Start the main Tkinter event loop."""
        self._root.mainloop()