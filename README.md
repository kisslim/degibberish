# degibberish

fix encoding errors automatically by a (large?) language model.

this could (theoretically) even take an wrongly decoded string and give you natural output. (we just try encode then decode exhaustively.)

actually it works for any language model.

> [chardet](https://pypi.org/project/chardet/) use a hand-crafted method and only works for bytes.

i use an autoregressive one just because it is now well known.
