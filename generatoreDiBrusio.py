from datasets import load_from_disk
import time
import torch
import torchvision
from lib.IPPy.IPPy.operators import *
from constants import OUTPUT_DIR_NOISE, OUTPUT_DIR_BLUR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Caricamento del dataset
ds = load_from_disk(OUTPUT_DIR_BLUR)
transform = torchvision.transforms.ToPILImage()

def aggiungi_rumore(riga_dataset, livello_rumore: float = 0.005):
    # 1. Estrai l'immagine dalla riga del dataset (è la chiave "image")
    immagine_pil = riga_dataset["image"]
    
    # Prevenzione crash: assicurati che sia in formato RGB (in caso di immagini in scala di grigi)
    #if immagine_pil.mode != "RGB":
    #    immagine_pil = immagine_pil.convert("RGB")

    # 2. Converti in tensore e sposta su GPU (CUDA)
    tensore_img = torchvision.transforms.functional.pil_to_tensor(immagine_pil).to(DEVICE)
    tensore_normalizzato = tensore_img.float() / 255.0
    tensore_batch = tensore_normalizzato.unsqueeze(0)
    
    # 3. Genera e scala il rumore
    rumore_base = torch.randn_like(tensore_batch, device=DEVICE)
    rumore_scalato = (rumore_base / torch.norm(rumore_base)) * torch.norm(tensore_batch) * livello_rumore
    
    # 4. Somma il rumore all'immagine e rimuovi la dimensione del batch
    immagine_rumorosa = tensore_batch + rumore_scalato  
    tensore_output = immagine_rumorosa.squeeze(0)
    
    # 5. Riporta il tensore ai valori standard dei pixel [0, 255] (Interi a 8 bit)
    tensore_finale_uint8 = (tensore_output * 255.0).clamp(0, 255).to(torch.uint8)
    
    # 6. Sposta il tensore sulla CPU (ToPILImage andrebbe in crash se restasse su CUDA) e converti
    immagine_finale_pil = transform(tensore_finale_uint8.cpu())
    
    # 7. Sostituisci l'immagine originale con quella elaborata nel dizionario
    riga_dataset["image"] = immagine_finale_pil
    
    # .map() richiede che venga restituito il dizionario aggiornato
    return riga_dataset

# Applica la funzione a tutto il dataset (con barra di avanzamento tqdm)
print("Avvio aggiunta rumore...")
t0 = time.time()
ds_procc = ds.map(aggiungi_rumore, desc="Aggiunta rumore")
totale = time.time() - t0
print(f"Rumore aggiunto in {totale:.1f}s ({totale/60:.1f} min)")

print("Salvataggio su disco...")
ds_procc.save_to_disk(OUTPUT_DIR_NOISE)
print(f"Dataset salvato in: {OUTPUT_DIR_NOISE}")