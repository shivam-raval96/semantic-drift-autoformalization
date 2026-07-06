# Example stories: one theme, several implications

The counterpart to [examples.md](examples.md): there, one implication
wears every theme; here, the `tea` theme informalizes several
different implications. The pairs are chosen for their shapes —
bare-variable sides, repeated variables, deep nesting — not for
their ETP numbering, and truth is irrelevant: every (E, F) pair is
rendered identically, as an unanswered question.

As in examples.md, the formal statements appear only in this
documentation, never in the story texts. Each story was generated
with:

```sh
python3 storyform.py "<E>" "<F>" --theme tea
```

## 1. `x = x ∘ x` ⇒ `x ∘ y = y ∘ x`

*The habit's left side is a bare variable, so it renders as a single procedure compared against the starting tea itself.*

In a certain mountain teahouse, the tea master follows one unbreakable habit. Take any tea at all — call it jasmine.

He pours jasmine over the leaves of jasmine and sets the result aside as Brew 1.

However he chooses his starting tea, jasmine itself and Brew 1 always taste exactly the same. That is simply how this teahouse works, without exception.

One evening his newest server wonders about something. Take any two teas — again call the first jasmine and the second oolong. Pour jasmine over the leaves of oolong and set the result aside as Brew 1. Separately, pour oolong over the leaves of jasmine and call the result Brew 2.

In this teahouse, must Brew 1 and Brew 2 always taste the same?

## 2. `x ∘ (y ∘ z) = (x ∘ y) ∘ z` ⇒ `x ∘ x = x`

*Three variables and mirrored nesting: the two procedures differ only in which brew is poured over which leaves.*

In a certain mountain teahouse, the tea master follows one unbreakable habit. Take any three teas at all — call the first jasmine, the second oolong, and the third rooibos. He runs two procedures side by side.

In the first, he pours oolong over the leaves of rooibos and sets the result aside as Brew 1; then he pours jasmine over the leaves of Brew 1 and calls the result Brew 2.

In the second, he pours jasmine over the leaves of oolong and calls that Brew 3; then he pours Brew 3 over the leaves of rooibos and sets the result aside as Brew 4.

However he chooses his three starting teas, Brew 2 and Brew 4 always taste exactly the same. That is simply how this teahouse works, without exception.

One evening his newest server wonders about something. Take any tea — again call it jasmine. Pour jasmine over the leaves of jasmine and set the result aside as Brew 1.

In this teahouse, must Brew 1 and jasmine itself always taste the same?

## 3. `x ∘ (y ∘ (z ∘ w)) = ((x ∘ y) ∘ z) ∘ w` ⇒ `x = y ∘ x`

*Four variables and maximal nesting on both sides — the longest procedures a four-operation law can produce.*

In a certain mountain teahouse, the tea master follows one unbreakable habit. Take any four teas at all — call the first jasmine, the second oolong, the third rooibos, and the fourth sencha. He runs two procedures side by side.

In the first, he pours rooibos over the leaves of sencha and sets the result aside as Brew 1; then he pours oolong over the leaves of Brew 1 and calls the result Brew 2; then he pours jasmine over the leaves of Brew 2 and calls that Brew 3.

In the second, he pours jasmine over the leaves of oolong and sets the result aside as Brew 4; then he pours Brew 4 over the leaves of rooibos and calls the result Brew 5; then he pours Brew 5 over the leaves of sencha and calls that Brew 6.

However he chooses his four starting teas, Brew 3 and Brew 6 always taste exactly the same. That is simply how this teahouse works, without exception.

One evening his newest server wonders about something. Take any two teas — again call the first jasmine and the second oolong. Pour oolong over the leaves of jasmine and set the result aside as Brew 1.

In this teahouse, must jasmine itself and Brew 1 always taste the same?

## 4. `x = y` ⇒ `x ∘ y = y ∘ x`

*Both sides of the habit are bare variables, so the habit has no procedures at all, only the regularity itself.*

In a certain mountain teahouse, the tea master follows one unbreakable habit. Take any two teas at all — call the first jasmine and the second oolong.

However he chooses his two starting teas, jasmine itself and oolong itself always taste exactly the same. That is simply how this teahouse works, without exception.

One evening his newest server wonders about something. Take any two teas — again call the first jasmine and the second oolong. Pour jasmine over the leaves of oolong and set the result aside as Brew 1. Separately, pour oolong over the leaves of jasmine and call the result Brew 2.

In this teahouse, must Brew 1 and Brew 2 always taste the same?

## 5. `(x ∘ x) ∘ x = x ∘ (x ∘ x)` ⇒ `x = x ∘ x`

*A single variable used repeatedly: every step pours some brew over the same tea's leaves, and only the nesting shape differs.*

In a certain mountain teahouse, the tea master follows one unbreakable habit. Take any tea at all — call it jasmine. He runs two procedures side by side.

In the first, he pours jasmine over the leaves of jasmine and sets the result aside as Brew 1; then he pours Brew 1 over the leaves of jasmine and calls the result Brew 2.

In the second, he pours jasmine over the leaves of jasmine and calls that Brew 3; then he pours jasmine over the leaves of Brew 3 and sets the result aside as Brew 4.

However he chooses his starting tea, Brew 2 and Brew 4 always taste exactly the same. That is simply how this teahouse works, without exception.

One evening his newest server wonders about something. Take any tea — again call it jasmine. Pour jasmine over the leaves of jasmine and set the result aside as Brew 1.

In this teahouse, must jasmine itself and Brew 1 always taste the same?
