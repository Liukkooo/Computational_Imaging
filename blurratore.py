from datasets import load_dataset
import time
import torch
import torchvision
from lib.IPPy.IPPy.operators import *
from constants import OUTPUT_DIR_BLUR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ds = load_dataset("benjamin-paine/imagenet-1k-256x256")
b = Blurring(img_shape=[3,256,256],kernel_type="gaussian", kernel_size=9,kernel_variance=2**2)

transform = torchvision.transforms.ToPILImage()

def aggiungi_blur(riga_dataset):
    immagine_pil = riga_dataset["image"]
    # Load GT image
    tensore_img = torchvision.transforms.functional.pil_to_tensor(immagine_pil).to(DEVICE)

    # Convertiamo in Float e portiamo nel range [0.0, 1.0]
    tensore_normalizzato = tensore_img.float() / 255.0
    tensore_batch = tensore_normalizzato.unsqueeze(0)  # Aggiungiamo il batch per la convoluzione -> [1, 3, 256, 256]

    # --- 2. ELABORAZIONE ---
    # Applichiamo il tuo operatore di blurring
    tensore_blurrato = b._matvec(tensore_normalizzato)
   
    # --- 3. RITORNO (Denormalizzazione) ---
    # Rimuoviamo il batch per tornare a [3, 256, 256]
    output_blur = tensore_blurrato.squeeze(0)

    # Riportiamo nel range 0-255 e riconvertiamo in Byte (uint8)
    # Usiamo .clamp(0, 255) per sicurezza, per evitare che piccoli errori di arrotondamento
    # portino pixel a -0.001 o 255.002, rompendo la conversione in uint8
    tensore_finale_uint8 = (output_blur * 255.0).clamp(0, 255).to(torch.uint8)

    immagine_finale_pil = transform(tensore_finale_uint8.cpu())
    
    # 7. Sostituisci l'immagine originale con quella elaborata nel dizionario
    riga_dataset["image"] = immagine_finale_pil
    
    # .map() richiede che venga restituito il dizionario aggiornato
    return riga_dataset

# Applica la funzione a tutto il dataset (con barra di avanzamento tqdm)
print("Avvio blurring...")
t0 = time.time()
ds_procc = ds.map(aggiungi_blur, desc="Blurring immagini")
totale = time.time() - t0
print(f"Blurring completato in {totale:.1f}s ({totale/60:.1f} min)")

print("Salvataggio su disco...")
ds_procc.save_to_disk(OUTPUT_DIR_BLUR)
print(f"Dataset salvato in: {OUTPUT_DIR_BLUR}")