# paralinguistics-pipeline

Implementacija pipeline-a, opisanega v diplomski nalogi *"Nadgradnja slovenskega TTS sistema s parajezikovnimi zmožnostmi"*.

Pipeline omogoča dodajanje parajezikovnih oznak in zaznavanje emocij v transkripcijah zvočnih posnetkov, podanih v mapi `audio/`. Zvočni posnetki v tej mapi naj imajo končnico `.flac`.

ASR (Automatic Speech Recognition) proces ni del pipeline-a, zato je v korenskem imeniku pričakovana naslednja datoteka, formatirana skladno s konvencijami Kaldi:

- `text` - vsebuje besedilne transkripcije zvočnih posnetkov. Identifikator vsake transkripcije naj bo enak imenu pripadajoče zvočne datoteke v `audio/`. Pričakovano je da so vsi posnetki 24kHz.

Vmesni in končni rezultati se shranjujejo v mapo `outputs/`.

Repozitorij vključuje tudi kodo in modele iz drugih odprtokodnih repozitorijev, med njimi:
- PretrainedSED: https://github.com/fschmid56/PretrainedSED
- forced_alignment: https://github.com/jan3zk/forced_alignment

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
