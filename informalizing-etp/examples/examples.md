# Example stories

One implication rendered under every theme. The implication is the
worked example from [CLAUDE.md](CLAUDE.md):

- **E** (the habit): `x ∘ y = (y ∘ y) ∘ x` — ETP equation E387
- **F** (the question): `x ∘ y = y ∘ x` — ETP equation E43

The formal pair appears only here, in the documentation; the story
texts below contain no trace of it. All four stories share the same
skeleton — only the theme's vocabulary changes. Argument order,
step structure, and intermediate numbering are identical, which is
what makes each story mechanically invertible back to the same pair
of term trees. Hash-based auto-selection would pick the `tea`
theme for this pair; every theme is forced explicitly here.

This file was generated with:

```sh
python3 storyform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x" --theme <key>
```

## Theme: `graft`

In a certain hillside orchard, the gardener follows one unbreakable habit. Take any two cuttings at all — call the first quince and the second medlar. She runs two procedures side by side.

In the first, she grafts quince onto medlar and sets the result aside as Graft 1.

In the second, she grafts medlar onto medlar and calls the result Graft 2; then she grafts Graft 2 onto quince and calls that Graft 3.

However she chooses her two starting cuttings, Graft 1 and Graft 3 always grow into exactly the same plant. That is simply how this orchard works, without exception.

One spring her neighbor wonders about something. Take any two cuttings — again call the first quince and the second medlar. Graft quince onto medlar and set the result aside as Graft 1. Separately, graft medlar onto quince and call the result Graft 2.

In this orchard, must Graft 1 and Graft 2 always grow into the same plant?

## Theme: `paint`

In a certain paint workshop, the colorist follows one unbreakable habit. Take any two pigments at all — call the first crimson and the second ochre. She runs two procedures side by side.

In the first, she pours crimson into ochre and sets the result aside as Batch 1.

In the second, she pours ochre into ochre and calls the result Batch 2; then she pours Batch 2 into crimson and calls that Batch 3.

However she chooses her two starting pigments, Batch 1 and Batch 3 always come out the exact same color. That is simply how this workshop works, without exception.

One morning her apprentice wonders about something. Take any two pigments — again call the first crimson and the second ochre. Pour crimson into ochre and set the result aside as Batch 1. Separately, pour ochre into crimson and call the result Batch 2.

In this workshop, must Batch 1 and Batch 2 always come out the same color?

## Theme: `signal`

At a certain mountaintop relay station, the operator follows one unbreakable habit. Take any two signals at all — call the first whistle and the second hum. He runs two procedures side by side.

In the first, he feeds whistle through hum and sets the result aside as Relay 1.

In the second, he feeds hum through hum and calls the result Relay 2; then he feeds Relay 2 through whistle and calls that Relay 3.

However he chooses his two starting signals, Relay 1 and Relay 3 always sound exactly alike. That is simply how this station works, without exception.

One night his trainee wonders about something. Take any two signals — again call the first whistle and the second hum. Feed whistle through hum and set the result aside as Relay 1. Separately, feed hum through whistle and call the result Relay 2.

At this station, must Relay 1 and Relay 2 always sound exactly alike?

## Theme: `tea`

In a certain mountain teahouse, the tea master follows one unbreakable habit. Take any two teas at all — call the first jasmine and the second oolong. He runs two procedures side by side.

In the first, he pours jasmine over the leaves of oolong and sets the result aside as Brew 1.

In the second, he pours oolong over the leaves of oolong and calls the result Brew 2; then he pours Brew 2 over the leaves of jasmine and calls that Brew 3.

However he chooses his two starting teas, Brew 1 and Brew 3 always taste exactly the same. That is simply how this teahouse works, without exception.

One evening his newest server wonders about something. Take any two teas — again call the first jasmine and the second oolong. Pour jasmine over the leaves of oolong and set the result aside as Brew 1. Separately, pour oolong over the leaves of jasmine and call the result Brew 2.

In this teahouse, must Brew 1 and Brew 2 always taste the same?
