import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from TRITON_SWMM_toolkit import run_model
from pathlib import Path
import yaml
from TRITON_SWMM_toolkit.config import SimulationConfig


def launch_gui():
    # Use TkinterDnD for drag-and-drop
    root = TkinterDnD.Tk()
    root.title("TRITON SWMM Toolkit")
    root.geometry("600x300")

    # ------------------------------
    # Config file input
    # ------------------------------
    tk.Label(root, text="Config File:").pack(pady=(10, 0))
    entry_path = tk.Entry(root, width=60)
    entry_path.pack(padx=10, pady=5)

    def browse_file():
        path = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if path:
            entry_path.delete(0, tk.END)
            entry_path.insert(0, path)

    tk.Button(root, text="Browse", command=browse_file).pack(pady=(0, 5))

    # Enable drag-and-drop
    def drop(event):
        files = root.tk.splitlist(event.data)
        if files:
            entry_path.delete(0, tk.END)
            entry_path.insert(0, files[0])  # only take first file

    entry_path.drop_target_register(DND_FILES)  # type: ignore
    entry_path.dnd_bind("<<Drop>>", drop)  # type: ignore

    # ------------------------------
    # Additional simulation options
    # ------------------------------
    verbose_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Verbose", variable=verbose_var).pack(pady=5)

    # Example: numeric setting (e.g., timestep)
    tk.Label(root, text="Time Step (minutes):").pack(pady=(10, 0))
    entry_timestep = tk.Entry(root, width=10)
    entry_timestep.insert(0, "5")
    entry_timestep.pack()

    # ------------------------------
    # Run button
    # ------------------------------
    def run():
        config_path = entry_path.get()
        verbose = verbose_var.get()
        timestep = entry_timestep.get()

        if not config_path:
            messagebox.showwarning("Missing file", "Please select a config file!")
            return
        try:
            # Convert timestep to int
            timestep = int(timestep)
            # Call core logic
            run_model(config_path=config_path, verbose=verbose, timestep=timestep)
            messagebox.showinfo("Done", "Simulation finished successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(root, text="Run Simulation", command=run, bg="green", fg="white").pack(
        pady=15
    )

    root.mainloop()


# %% GUI for creating a config file


# Tooltip class
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = (
            self.widget.bbox("insert") if self.widget.winfo_ismapped() else (0, 0, 0, 0)
        )
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TRITON-SWMM Config Editor")
        self.root.geometry("400x360")

        self.config_path: Path | None = None

        # Step 1: Let user select or create a config file
        tk.Label(
            self.root,
            text="Select or create config file:",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 0))
        self.config_var = tk.StringVar()
        config_frame = tk.Frame(self.root)
        config_frame.pack(fill="x", padx=10)
        tk.Entry(config_frame, textvariable=self.config_var).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(config_frame, text="Browse", command=self.browse_config_file).pack(
            side="left", padx=5
        )

        # Frame for the rest of the fields (hidden until config file is chosen)
        self.fields_frame = tk.Frame(self.root)
        self.fields_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.fields_frame.pack_forget()  # hide initially

        # Variables for config fields
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.timestep_var = tk.IntVar(value=5)
        self.verbose_var = tk.BooleanVar(value=False)

        self._build_fields()

    def browse_config_file(self):
        # Use asksaveasfilename so the user can type a new file
        file = filedialog.asksaveasfilename(
            title="Select or create config file",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        if file:
            self.config_var.set(file)
            self.config_path = Path(file)
            self._populate_fields_from_file()

    def _populate_fields_from_file(self):
        """Load YAML if exists, else leave fields blank."""
        data = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to load config: {e}")

        self.input_var.set(data.get("input_file", ""))
        self.output_var.set(data.get("output_dir", ""))
        self.timestep_var.set(data.get("timestep", 5))
        self.verbose_var.set(data.get("verbose", False))

        # Show the fields frame
        self.fields_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_fields(self):
        # Input file
        self._add_file_field(
            label="SWMM Input File:",
            var=self.input_var,
            browse_command=self.browse_input,
            tooltip=SimulationConfig.model_fields["input_file"].description,
        )

        # Output directory
        self._add_file_field(
            label="Output Directory:",
            var=self.output_var,
            browse_command=self.browse_output,
            tooltip=SimulationConfig.model_fields["output_dir"].description,
        )

        # Timestep
        tk.Label(self.fields_frame, text="Timestep (minutes):").pack(
            anchor="w", pady=(10, 0)
        )
        entry = tk.Entry(self.fields_frame, textvariable=self.timestep_var)
        entry.pack(fill="x")
        ToolTip(entry, SimulationConfig.model_fields["timestep"].description)

        # Verbose
        chk = tk.Checkbutton(
            self.fields_frame, text="Verbose", variable=self.verbose_var
        )
        chk.pack(anchor="w", pady=10)
        ToolTip(chk, SimulationConfig.model_fields["verbose"].description)

        # Save button
        tk.Button(self.fields_frame, text="Save Config", command=self.save_config).pack(
            pady=10
        )

    def _add_file_field(
        self, label: str, var: tk.Variable, browse_command, tooltip: str
    ):
        tk.Label(self.fields_frame, text=label).pack(anchor="w", pady=(10, 0))
        frame = tk.Frame(self.fields_frame)
        frame.pack(fill="x")
        entry = tk.Entry(frame, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)
        ToolTip(entry, tooltip)
        tk.Button(frame, text="Browse", command=browse_command).pack(
            side="left", padx=5
        )

    def browse_input(self):
        file = filedialog.askopenfilename(
            title="Select SWMM Input File",
            filetypes=[("INP files", "*.inp"), ("All files", "*.*")],
        )
        if file:
            self.input_var.set(file)

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_var.set(folder)

    def save_config(self):
        if not self.config_path:
            messagebox.showerror("Error", "Please select a config file path first.")
            return
        try:
            # Validate using Pydantic before saving
            config = SimulationConfig(
                input_file=Path(self.input_var.get()),
                output_dir=Path(self.output_var.get()),
                timestep=self.timestep_var.get(),
                verbose=self.verbose_var.get(),
            )
            with open(self.config_path, "w") as f:
                yaml.safe_dump(config.model_dump(), f)
            messagebox.showinfo("Success", f"Config saved to {self.config_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


# %%
if __name__ == "__main__":
    gui = ConfigGUI()
    gui.run()
