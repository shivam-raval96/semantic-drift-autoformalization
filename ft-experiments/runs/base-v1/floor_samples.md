# Floor sanity samples — base runs, 10 wrong + 10 unparseable per model

## 1B

### wrong

**evaluation_hard_0017** · story · ops 6 · finish `stop`
- reference: `ASSUME: op(x, x) = op(x, y)` / `ASK: op(x, y) = op(op(op(x, x), z), y)`
```
ASSUME: op(jasmine, op(jasmine, oolong)) = oolong
ASK: op(jasmine, op(oolong, jasmine)) = op(oolong, jasmine)
```
**evaluation_extra_hard_0110** · story · ops 7 · finish `stop`
- reference: `ASSUME: op(x, y) = op(x, op(x, x))` / `ASK: op(x, y) = op(x, op(y, op(z, y)))`
```
ASSUME: op(hum, op(hum, whistle)) = whistle
ASK: op(hum, whistle) = op(whistle, hum)
```
**evaluation_extra_hard_0064** · story · ops 7 · finish `stop`
- reference: `ASSUME: op(x, y) = op(y, op(y, y))` / `ASK: op(x, y) = op(y, op(x, op(z, y)))`
```
ASSUME: op(quince, medlar) = op(medlar, quince)
ASK: op(medlar, op(quince, medlar)) = op(medlar, quince)
```
**evaluation_normal_0131** · story · ops 8 · finish `stop`
- reference: `ASSUME: x = op(op(y, x), op(z, op(y, z)))` / `ASK: x = op(op(op(y, x), y), op(y, z))`
```
ASSUME: op(jasmine, op(oolong, rooibos)) = rooibos
ASK: op(oolong, op(oolong, rooibos)) = jasmine
```
**evaluation_hard_0111** · story · ops 7 · finish `stop`
- reference: `ASSUME: x = op(op(y, x), op(x, z))` / `ASK: x = op(op(x, y), op(op(z, z), z))`
```
ASSUME: op(jasmine, oolong) = op(oolong, jasmine)
ASK: op(jasmine, op(oolong, jasmine)) = op(oolong, jasmine)
```
**evaluation_order5_0064** · literal · ops 10 · finish `stop`
- reference: `ASSUME: x = op(op(y, op(z, op(w, op(y, u)))), u)` / `ASK: x = op(op(y, op(y, y)), op(y, op(z, x)))`
```
ASSUME: op(y, op(x, u)) = x
ASK: op(y, op(x, u)) = op(x, op(y, u))
```
**evaluation_extra_hard_0122** · literal · ops 7 · finish `stop`
- reference: `ASSUME: op(x, y) = op(y, op(y, y))` / `ASK: op(x, y) = op(x, op(x, op(z, y)))`
```
ASSUME: op(x, op(x, y)) = y
ASK: op(x, y) = op(y, x)
```
**evaluation_order5_0086** · literal · ops 10 · finish `stop`
- reference: `ASSUME: x = op(y, op(z, op(op(w, op(y, z)), u)))` / `ASK: x = op(op(op(y, z), op(w, x)), op(w, y))`
```
ASSUME: op(y, op(x, z)) = x
ASK: op(y, op(w, x)) = op(w, op(z, x))
```
**evaluation_order5_0034** · literal · ops 10 · finish `stop`
- reference: `ASSUME: x = op(op(op(op(y, z), op(w, w)), x), y)` / `ASK: x = op(op(y, z), op(op(x, op(z, z)), z))`
```
ASSUME: op(y, op(x, z)) = x
ASK: op(x, op(y, z)) = x
```
**evaluation_normal_0088** · literal · ops 8 · finish `stop`
- reference: `ASSUME: x = op(x, op(op(y, op(z, z)), w))` / `ASK: op(x, y) = op(op(x, z), op(x, y))`
```
ASSUME: op(x, op(x, y)) = y
ASK: op(x, y) = op(op(x, y), x)
```

### unparseable

**evaluation_normal_0018** · story · ops 5 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(x, op(op(y, op(z, z)), z))` / `ASK: x = op(x, x)`
```
ASSUME: op(crimson, op(crimson, Batch 1)) = Batch 1
ASK: op(crimson, Batch 1) = op(Batch 1, crimson)
```
**evaluation_order5_0080** · story · ops 10 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(y, op(op(y, x), op(z, op(w, y))))` / `ASK: x = op(y, op(op(op(x, op(x, x)), y), y))`
```
ASSUME: op(crimson, Batch 1) = Batch 1
ASK: op(crimson, Batch 2) = op(Batch 1, ochre)
```
**evaluation_hard_0189** · story · ops 8 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(op(y, z), op(op(w, w), x))` / `ASK: x = op(x, op(y, op(op(x, z), x)))`
```
ASSUME: op(crimson, Batch 1) = Batch 4
ASK: op(crimson, Batch 4) = op(crimson, Batch 1)
```
**evaluation_order5_0193** · story · ops 10 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(x, op(y, op(z, op(op(z, w), y))))` / `ASK: x = op(y, op(op(z, op(z, op(w, x))), y))`
```
ASSUME: op(whistle, hum) = Relay 1
ASK: op(whistle, Relay 1) = op(hum, whistle)
```
**evaluation_hard_0158** · story · ops 8 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(y, op(x, op(op(x, z), w)))` / `ASK: op(x, y) = op(op(x, x), op(z, x))`
```
ASSUME: op(whistle, hum) = Relay 1
ASK: op(whistle, Relay 1) = op(hum, Relay 1)
```
**evaluation_hard_0152** · literal · ops 8 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(y, op(z, op(x, op(w, u))))` / `ASK: x = op(op(y, op(x, op(x, x))), z)`
```
ASSUME: op(w, u) = Value 1
ASK: op(x, y) = op(z, Value 2)
```
**evaluation_order5_0088** · literal · ops 10 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(op(op(op(y, x), op(z, x)), w), z)` / `ASK: x = op(op(op(y, x), op(z, x)), op(x, z))`
```
ASSUME: op(y, x) = op(z, x) = op(Value 1, x) = x
ASK: op(y, x) = op(z, x)
```
**evaluation_order5_0010** · literal · ops 10 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(x, op(y, op(z, op(z, op(z, w)))))` / `ASK: x = op(x, op(op(y, op(op(z, y), w)), y))`
```
ASSUME: op(z, w) = Value 1
ASK: op(z, Value 1) = op(z, w)
```
**evaluation_order5_0046** · literal · ops 10 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(y, op(op(z, op(op(w, x), z)), z))` / `ASK: x = op(op(y, z), op(op(x, op(x, z)), y))`
```
ASSUME: op(w, x) = op(z, y) = op(z, Value 1) = op(z, Value 2) = op(z, Value 3) = op(z, Value 4) = op(z, Value 5)
ASK: op(x, y) = op(y, z) = op(x, Value 1) = op(x, Value 2) = op(x, Value 3) = op(x, Value 4) = op(x, Value 5)
```
**evaluation_order5_0172** · literal · ops 10 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(y, op(op(op(op(z, w), z), x), z))` / `ASK: x = op(op(y, op(x, x)), op(op(y, z), w))`
```
ASSUME: op(z, w) = Value 1
ASK: op(x, x) = op(y, z)
```
## 8B

### wrong

**evaluation_order5_0045** · story · ops 10 · finish `stop`
- reference: `ASSUME: x = op(op(op(op(y, op(z, w)), x), y), x)` / `ASK: x = op(y, op(op(op(op(z, z), x), x), w))`
```
ASSUME: op(op(op(op(rooibos, sencha), oolong), jasmine), oolong) = jasmine
ASK: op(op(op(op(rooibos, rooibos), jasmine), jasmine), jasmine) = op(op(op(op(rooibos, rooibos), jasmine), jasmine), sencha)
```
**evaluation_extra_hard_0158** · story · ops 7 · finish `stop`
- reference: `ASSUME: op(x, x) = op(y, op(x, z))` / `ASK: op(x, x) = op(op(x, x), op(y, z))`
```
ASSUME: op(jasmine, op(jasmine, rooibos)) = op(jasmine, oolong)
ASK: op(jasmine, op(jasmine, jasmine)) = op(op(jasmine, jasmine), op(oolong, rooibos))
```
**evaluation_extra_hard_0075** · story · ops 8 · finish `stop`
- reference: `ASSUME: x = op(op(y, x), op(x, op(z, w)))` / `ASK: op(x, op(x, y)) = op(op(x, x), y)`
```
ASSUME: op(op(whistle, op(hum, buzz)), op(op(whistle, op(chirp, buzz)), op(hum, op(chirp, buzz)))) = whistle
ASK: op(op(whistle, op(whistle, op(whistle, hum))), op(op(whistle, op(whistle, op(whistle, hum))), hum)) = op(op(whistle, op(whistle, hum)), op(whistle, hum))
```
**evaluation_order5_0166** · story · ops 10 · finish `stop`
- reference: `ASSUME: x = op(y, op(op(z, op(x, op(y, y))), y))` / `ASK: x = op(y, op(op(y, z), op(x, op(w, x))))`
```
ASSUME: op(teal, op(op(crimson, op(ochre, teal)), ochre)) = op(crimson, op(op(crimson, op(ochre, teal)), ochre))
ASK: op(crimson, op(op(indigo, crimson), op(ochre, teal))) = op(op(ochre, teal), op(crimson, op(indigo, crimson)))
```
**evaluation_normal_0190** · story · ops 8 · finish `stop`
- reference: `ASSUME: op(x, y) = op(z, op(x, op(x, z)))` / `ASK: op(x, op(y, z)) = op(op(z, x), x)`
```
ASSUME: op(quince, op(damson, op(quince, op(damson, op(quince, damson))))) = op(quince, damson)
ASK: op(quince, op(medlar, damson)) = op(damson, quince)
```
**evaluation_hard_0099** · literal · ops 7 · finish `stop`
- reference: `ASSUME: x = op(op(y, x), op(x, z))` / `ASK: x = op(y, op(op(op(y, x), x), y))`
```
ASSUME: op(y, op(x, z)) = op(op(y, x), z)
ASK: op(op(op(y, x), x), y) = op(y, op(op(y, x), x))
```
**evaluation_hard_0002** · literal · ops 4 · finish `stop`
- reference: `ASSUME: x = op(y, op(z, z))` / `ASK: x = op(op(y, z), z)`
```
ASSUME: op(z, op(y, op(z, z))) = x
ASK: op(op(y, z), z) = op(z, op(y, z))
```
**evaluation_order5_0092** · literal · ops 10 · finish `stop`
- reference: `ASSUME: x = op(op(op(y, z), w), op(w, op(u, y)))` / `ASK: x = op(op(y, op(x, op(z, y))), op(x, w))`
```
ASSUME: op(op(y, z), w) = op(op(u, y), op(w, op(y, z)))
ASK: op(x, op(z, y)) = op(op(x, op(z, y)), op(w, op(y, op(x, op(z, y)))))
```
**evaluation_extra_hard_0087** · literal · ops 8 · finish `stop`
- reference: `ASSUME: x = op(op(y, x), op(x, op(z, w)))` / `ASK: op(op(x, x), y) = op(op(x, y), x)`
```
ASSUME: op(x, op(z, w)) = op(op(y, x), op(x, op(z, w)))
ASK: op(op(x, x), y) = op(op(x, op(x, x)), y)
```
**evaluation_normal_0018** · literal · ops 5 · finish `stop`
- reference: `ASSUME: x = op(x, op(op(y, op(z, z)), z))` / `ASK: x = op(x, x)`
```
ASSUME: op(x, op(y, op(z, z))) = op(op(y, op(z, z)), op(x, z))
ASK: op(x, op(x, x)) = x
```

### unparseable

**evaluation_order5_0158** · story · ops 10 · finish `stop` · parse error: `expected ',', got '='`
- reference: `ASSUME: x = op(y, op(z, op(op(y, y), op(w, w))))` / `ASK: x = op(y, op(op(op(y, op(x, z)), y), w))`
```
ASSUME: op(op(op(op(oolong, sencha), op(rooibos, op(op(oolong, sencha), op(rooibos, op(oolong, sencha))))), oolong) = sencha
ASK: op(op(op(op(jasmine, rooibos), op(oolong, op(jasmine, rooibos))), oolong), oolong) = jasmine
```
**evaluation_normal_0093** · story · ops 8 · finish `stop` · parse error: `expected ')', got None`
- reference: `ASSUME: op(x, x) = op(op(y, x), op(z, z))` / `ASK: op(x, y) = op(op(z, x), op(w, u))`
```
ASSUME: op(crimson, op(crimson, crimson)) = op(teal, op(ochre, op(teal, op(teal, op(ochre, crimson)))))
ASK: op(crimson, op(ochre, crimson)) = op(teal, op(saffron, op(crimson, op(teal, op(saffron, op(indigo, saffron)))))
```
**evaluation_hard_0044** · story · ops 6 · finish `stop` · parse error: `expected '=', got None`
- reference: `ASSUME: x = op(y, op(x, op(x, z)))` / `ASK: x = op(op(y, op(x, z)), x)`
```
ASSUME: op(quince, op(quince, op(quince, op(medlar, op(quince, op(quince, damson))))))
ASK: op(quince, op(op(medlar, op(quince, damson)), quince)) = op(quince, op(quince, op(medlar, op(quince, damson))))
```
**evaluation_order5_0126** · story · ops 10 · finish `stop` · parse error: `unexpected character '1'`
- reference: `ASSUME: x = op(op(op(op(y, op(x, x)), x), y), z)` / `ASK: x = op(op(op(x, y), x), op(op(z, z), y))`
```
ASSUME: op(op(op(op(op(whistle, whistle), Relay 1), whistle), Relay 2), hum) = whistle
ASK: op(op(op(op(Relay 2, Relay 4), hum), chirp), hum) = op(op(whistle, hum), op(chirp, chirp))
```
**evaluation_normal_0073** · story · ops 8 · finish `stop` · parse error: `expected ',', got ')'`
- reference: `ASSUME: x = op(op(y, y), op(z, op(w, x)))` / `ASK: x = op(y, op(y, op(x, op(z, w))))`
```
ASSUME: op(op(op(op(crimson, indigo), op(teal, op(crimson, indigo))), op(ochre, op(op(crimson, indigo), op(teal, op(crimson, indigo))))), op(ochre, op(op(crimson, indigo), op(teal, op(crimson, indigo))))) = crimson
ASK: op(op(op(op(crimson, indigo), op(teal, indigo)), op(ochre, op(op(crimson, indigo), op(teal, indigo))))), op(op(op(op(ochre, indigo), op(teal, indigo)), op(crimson, op(op(ochre, ind …[637 chars total]
```
**evaluation_extra_hard_0126** · literal · ops 7 · finish `stop` · parse error: `expected '=', got ')'`
- reference: `ASSUME: op(x, y) = op(y, op(y, y))` / `ASK: op(x, y) = op(x, op(z, op(x, y)))`
```
ASSUME: op(x, op(y, op(y, op(y, op(y, y))))) = op(x, op(y, op(y, op(y, y))))
ASK: op(x, op(y, op(z, op(x, op(x, op(z, op(x, op(y, op(y, op(y, op(y, y)))))))))))) = op(x, op(y, op(y, op(y, op(y, y))))))
```
**evaluation_normal_0082** · literal · ops 8 · finish `stop` · parse error: `trailing tokens in 'op(y, z) = op(y, y), op(op(y, z), op(y, y)) = x'`
- reference: `ASSUME: x = op(x, op(op(y, z), op(y, y)))` / `ASK: op(x, x) = op(op(x, y), op(z, x))`
```
ASSUME: op(y, z) = op(y, y), op(op(y, z), op(y, y)) = x
ASK: op(x, x) = op(op(x, y), op(z, x))
```
**evaluation_extra_hard_0105** · literal · ops 8 · finish `stop` · parse error: `trailing tokens in 'op(y, y) = op(op(y, y), x) = op(op(op(y, y), x), z)'`
- reference: `ASSUME: x = op(op(op(y, y), x), op(x, z))` / `ASK: x = op(x, op(op(y, op(y, x)), z))`
```
ASSUME: op(y, y) = op(op(y, y), x) = op(op(op(y, y), x), z)
ASK: op(y, x) = op(y, op(y, x)) = op(x, op(y, op(y, x)))
```
**evaluation_order5_0078** · literal · ops 10 · finish `stop` · parse error: `expected ')', got ','`
- reference: `ASSUME: x = op(op(op(op(y, x), op(z, w)), x), y)` / `ASK: x = op(op(x, op(x, y)), op(op(x, y), x))`
```
ASSUME: op(op(y, x), op(op(z, w), op(op(op(y, x), op(z, w)), x), y))) = x
ASK: op(op(x, op(x, op(x, y))), op(op(x, op(x, op(x, y))), x), y)) = op(op(x, op(x, op(x, y))), op(op(x, op(x, op(x, y))), x), y))
```
**evaluation_order5_0134** · literal · ops 10 · finish `stop` · parse error: `expected '=', got ','`
- reference: `ASSUME: x = op(y, op(z, op(op(op(y, y), z), w)))` / `ASK: x = op(op(op(op(y, z), w), w), op(x, z))`
```
ASSUME: op(y, op(op(y, y), z)) = op(op(op(y, op(y, y)), z), w)
ASK: op(x, op(op(y, z), op(op(y, z), w))), op(op(y, z), op(op(y, z), w)))
```
