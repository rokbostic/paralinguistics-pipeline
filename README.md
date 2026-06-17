# paralinguistics-pipeline

Implementacija pipeline-a, opisanega v diplomski nalogi *"Nadgradnja slovenskega TTS sistema s parajezikovnimi zmožnostmi"*.

Pipeline omogoča dodajanje parajezikovnih oznak in zaznavanje emocij v transkripcijah zvočnih posnetkov, podanih v mapi `audio/`. Zvočni posnetki v tej mapi naj imajo končnico `.flac`. Pričakovano je da so vsi posnetki 24kHz.

ASR (Automatic Speech Recognition) proces ni del pipeline-a, zato je v korenskem imeniku pričakovana datoteka `text`, ki vsebuje besedilne transkripcije zvočnih posnetkov, formatirana skladno s konvencijami Kaldi. Identifikator vsake transkripcije naj bo enak imenu pripadajoče zvočne datoteke v `audio/`.

Vmesni in končni rezultati se shranjujejo v mapo `outputs/`.

Repozitorij vključuje tudi kodo in modele iz drugih odprtokodnih repozitorijev, med njimi:
- PretrainedSED: https://github.com/fschmid56/PretrainedSED
- forced_alignment: https://github.com/jan3zk/forced_alignment

Za zazavanje medmetov je potrebno priskrbeti model za ta namen v `reources/nemo-train-asr-char.nemo`.

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

- `outputs/text_{POSTOPEK}`
  Transkripcije anotirane z parajezikovnimi oznakami

- `outputs/emotions_{POSTOPEK}` 
  Zaznane emocije za posamezne zvočne posnetke


Vmesni izhodi so:

- `outputs/medmet`
  Rezultat ASR postopka z NEMO modelom, ki je zmozen zaznati medmete. Oblika podobni datoteki `text`.

- `outputs/corpus`, `outputs/medmet_corpus`, `outputs/aligner`, `outputs/medmet_aligner`, `punctuate`
  Rezultati MFA

- `outputs/sed_{POSTOPEK}`
  Rezultati postopkov SED.