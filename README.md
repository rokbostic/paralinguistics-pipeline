# paralinguistics-pipeline

Implementacija pipeline-a, opisanega v diplomski nalogi *"Nadgradnja slovenskega TTS sistema s parajezikovnimi zmožnostmi"*.

Pipeline omogoča dodajanje parajezikovnih oznak in zaznavanje emocij v transkripcijah zvočnih posnetkov, podanih v mapi `audio/`. Zvočni posnetki v tej mapi naj imajo končnico `.flac`

ASR (Automatic Speech Recognition) proces ni del pipeline-a, zato sta kot vhod pričakovani naslednji datoteki:

- `text` - vsebuje besedilne transkripcije zvočnih posnetkov
- `wav.scp` - povezuje ID-je posnetkov z ustreznimi zvočnimi datotekami

Vmesni in končni rezultati se shranjujejo v mapo `outputs/`.

Repozitorij vključuje tudi kodo oziroma prilagoditve iz drugih odprtokodnih repozitorijev.

## Namestitev

Za nastavitev okolja zaženite:

```bash
./setup.sh
```

Skripta pripravi Conda okolje potrebno za naslednji korka.

## Zagon

Za zagon celotnega procesa zaženite:

```bash
./run.sh
```

## Rezultati

Glavna izhodna rezultata sta:

- `outputs/text` 
  Transkripcije z dodanimi parajezikovnimi anotacijami v obliki oznak (tagov)

- `outputs/emotions` 
  Zaznane emocije za posamezne zvočne posnetke
