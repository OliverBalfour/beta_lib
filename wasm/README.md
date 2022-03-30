
How to build the WASM pack:

```sh
cargo install wasm-pack
wasm-pack build --target web
```

Then you will get a `./pkg/beta.js` file with an `async init` default export function you must call and other named exports `pdf, cdf, ppf` that take `x, a, b`
