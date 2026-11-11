// Test helper: reads {"fn": name, "args": [...]} from stdin, calls ReviewLogic[fn], prints the JSON result.
const path = require("path");
const logic = require(path.join(__dirname, "..", "src", "docextract", "assets", "review_logic.js"));
let input = "";
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", () => {
  const { fn, args } = JSON.parse(input);
  const target = logic[fn];
  process.stdout.write(JSON.stringify(typeof target === "function" ? target(...args) : target));
});
