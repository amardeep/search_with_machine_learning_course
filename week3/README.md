# Self Assessment
## For classifying product names to categories:
See [Level 1 notebook](exploration/project-level1.ipynb) for details of experiment runs.

#### What precision (P@1) were you able to achieve?
91.5%

#### What fastText parameters did you use?
epochs=25, lr=1.0, ngrams=2, transform_product_names=True, min_products=50, max_depth=3

#### How did you transform the product names?
- Lowercase
- Remove punctuation
- Nltk work tokenizer
- Nltk snowball stemmer

#### How did you prune infrequent category labels, and how did that affect your precision?
Precision increased as more labels were pruned. Running with epochs=25, lr=1.0, ngrams=2, transform=True gave:

| min_products | 0      | 50    | 100   | 150   | 200   |
|--------------|--------|-------|-------|-------|-------|
| P@1          | .62    | .74   | .81   | .85   | .89   |
| num_cats     | 1952   | 520   | 269   | 168   | 113   |
| num_docs     | 115358 | 93126 | 75501 | 63188 | 53654 |


#### How did you prune the category tree, and how did that affect your precision?
Tried for different depth from root. Obviously, when depth = 1, everything is
in one category, so precision is 100%. The tradeoff is we are getting increasingly
coarser category mappings.

With min_products=50, there were about 520 categories. Running with epochs=25, lr=1.0, ngrams=2, transform=True, min_products=50 gave:

| max_depth | 1   | 2   | 3   | 4   |
|-----------|-----|-----|-----|-----|
| P@1       | 1.0 | .94 | .92 | .85 |
| num_cats  | 1   | 18  | 110 | 302 |

## For deriving synonyms from content:
See [Level 2 notebook](exploration/project-level2.ipynb) for details of experiment runs.

#### What 20 tokens did you use for evaluation?
##### Product Types
- Printer
- Vaccum
- Headphones
- Binoculars
- Phones
##### Brands
- Apple
- AMD
- Bosch
- Canon
- Skullcandy
##### Models
- FinePix
- Macbook
- Wii
- Aspire
- Ipad
##### Attributes
- 16GB
- Black
- Bonus
- Wireless
- USB

#### What fastText parameters did you use?
Tried different values for minCount parameter

#### How did you transform the product names?
Same as for category mapping:
- Lowercase
- Remove punctuation
- Nltk work tokenizer
- Nltk snowball stemmer

#### What threshold score did you use?
0.93

#### What synonyms did you obtain for those tokens?

| 1                       | 2                         | 3                              | 4                  | 5                | 6                 | 7                  | 8                    | 9                 | 10                   |
|-------------------------|---------------------------|--------------------------------|--------------------|------------------|-------------------|--------------------|----------------------|-------------------|----------------------|
| Printer (printer)       | printercopierscann: 0.850 | printercopierscannerfax: 0.848 | inkjet: 0.800      | copier: 0.798    | officejet: 0.790  | deskjet: 0.787     | dx4860ub32p: 0.779   | hl2170w: 0.776    | fax: 0.775           |
| Vaccum (vaccum)         | vacuum: 0.856             | roomba: 0.845                  | windtunnel: 0.841  | liftoff: 0.838   | vacmast: 0.834    | canist: 0.831      | vac: 0.828           | vax: 0.825        | bagless: 0.815       |
| Headphones (headphon)   | earbud: 0.872             | overtheear: 0.869              | overthehead: 0.804 | yurbud: 0.797    | earphon: 0.785    | noiseisol: 0.778   | behindtheneck: 0.765 | skullcandi: 0.764 | behindthehead: 0.752 |
| Binoculars (binocular)  | scope: 0.776              | barska: 0.768                  | bushnel: 0.739     | 2060: 0.704      | celestron: 0.687  | circular: 0.678    | pentax: 0.660        | altazimuth: 0.609 | 146mp: 0.606         |
| Phones (phone)          | huawei: 0.801             | kyocera: 0.778                 | gophon: 0.777      | phono: 0.776     | tmobil: 0.770     | razr: 0.748        | 4g: 0.743            | mobil: 0.742      | pantech: 0.741       |
| Apple (appl)            | appleâ: 0.849             | ipad: 0.784                    | ipod: 0.769        | iphon: 0.763     | ipodhd: 0.755     | 3rdgener: 0.732    | 4thgener: 0.721      | 6thgener: 0.720   | 3g3gs: 0.714         |
| AMD (amd)               | athlon: 0.892             | phenom: 0.883                  | tl60: 0.860        | quadcor: 0.851   | 3gb: 0.845        | am3: 0.839         | x6: 0.829            | turion: 0.826     | x4: 0.823            |
| Bosch (bosch)           | ascenta: 0.816            | integra: 0.783                 | 2378: 0.736        | woodlik: 0.715   | 67: 0.713         | tassimo: 0.711     | dishwash: 0.709      | 50lb: 0.694       | accubak: 0.692       |
| Canon (canon)           | eo: 0.814                 | 60d: 0.786                     | t2i: 0.773         | canoscan: 0.773  | 50d: 0.770        | pixma: 0.767       | 7d: 0.767            | sx230hs: 0.756    | 151mp: 0.752         |
| Skullcandy (skullcandi) | skull: 0.892              | earbud: 0.835                  | inkd: 0.831        | hesh: 0.784      | yurbud: 0.780     | gumi: 0.776        | bud: 0.774           | smokin: 0.767     | headphon: 0.764      |
| FinePix (finepix)       | fujifilm: 0.949           | z37: 0.917                     | xp20: 0.917        | z90: 0.906       | coolpix: 0.865    | w570: 0.855        | s210: 0.853          | w310: 0.852       | pl210: 0.848         |
| Macbook (macbook)       | bookendz: 0.783           | 154: 0.762                     | g4: 0.754          | powerbook: 0.737 | facebook: 0.684   | brenthaven: 0.678  | macbeth: 0.675       | lifebook: 0.671   | ultrabook: 0.668     |
| Wii (wii)               | nintendo: 0.869           | nintendog: 0.858               | ds: 0.828          | wwii: 0.769      | gamecub: 0.767    | 360: 0.721         | 3ds: 0.717           | challeng: 0.708   | paradis: 0.702       |
| Aspire (aspir)          | n270: 0.759               | 116: 0.739                     | ideapad: 0.718     | acer: 0.717      | i72670qm: 0.717   | obsidian: 0.716    | sapphir: 0.708       | qosmio: 0.700     | pentium: 0.696       |
| Ipad (ipad)             | appl: 0.784               | 3rd: 0.770                     | portfolio: 0.768   | folio: 0.752     | tribeca: 0.720    | ipadnetbook: 0.719 | bodhi: 0.717         | appleâ: 0.702     | ipodipad: 0.692      |
| 16GB (16gb)             | 32gb: 0.874               | 8gb: 0.838                     | 64gb: 0.802        | 4gb: 0.720       | memori: 0.715     | 6gb: 0.713         | jumpdriv: 0.695      | 12gb: 0.693       | 2gb: 0.684           |
| Black (black)           | blackwhit: 0.765          | blackpurpl: 0.764              | blacktop: 0.763    | blackr: 0.756    | blackorang: 0.754 | blacksilv: 0.741   | blackgold: 0.737     | whiteblack: 0.732 | redblack: 0.727      |
| Bonus (bonus)           | mysim: 0.744              | zombi: 0.740                   | vacat: 0.732       | zhu: 0.730       | poni: 0.720       | littlest: 0.715    | paradis: 0.715       | usaopoli: 0.707   | resid: 0.707         |
| Wireless (wireless)     | wirelessg: 0.859          | wirelessn: 0.770               | 80211b: 0.649      | 80211g: 0.646    | keyless: 0.639    | m305: 0.631        | m325: 0.627          | 80211n: 0.615     | router: 0.595        |
| USB (usb)               | 20firewir: 0.705          | drivest: 0.701                 | 20: 0.680          | 3020: 0.680      | drive: 0.669      | jumpdriv: 0.667    | firewir: 0.654       | 20esata: 0.653    | goflex: 0.648        |


## For integrating synonyms with search:
See [Level 3 notebook](exploration/project-level3.ipynb) for details of experiment runs.

#### How did you transform the product names (if different than previously)?
Same as earlier

#### What threshold score did you use?
.9

#### Were you able to find the additional results by matching synonyms?
Yes. All the example queries resulted in many more matches.

For `fabshell` it still results in only 8 matches, though with `fabshel`, it returns
many more matches. This happens as currently stemmer doesn't run on query.

## For classifying reviews:
This part was not done.

#### What precision (P@1) were you able to achieve?
#### What fastText parameters did you use?
#### How did you transform the review content?
#### What else did you try and learn?

# Log
### Example doc send to /annotate endpoint
```
{
   "longDescription":"Protect your phone and showcase your sense of style with this durable case.",
   "name":"Wireless Essentials - Senate Leather Case for Most Small Flip Phones - Black",
   "shortDescription":"From our expanded online assortment; compatible with most small flip phones; lambskin leather",
   "sku":"9036358"
}
```