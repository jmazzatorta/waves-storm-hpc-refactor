import os
import subprocess
import glob

def main():
    if not os.path.exists('generators'):
        print("Errore: cartella 'generators' non trovata. Lancia lo script da 'plots/'.")
        return

    # Trova gli script includendo la cartella nel percorso
    scripts = sorted(glob.glob('generators/*_plot.py'))
    
    if not scripts:
        print("Nessuno script trovato in generators/.")
        return

    print(f"Trovati {len(scripts)} script. Inizio generazione...\n")

    # Lancia ogni script stando fisicamente in plots/
    for script in scripts:
        print(f"Generazione in corso per: {script}")
        subprocess.run(['python', script])

    print("\nFinito! Tutti i grafici sono stati salvati correttamente nella directory plots/.")

if __name__ == "__main__":
    main()
