// @ts-check
import js from "@eslint/js";

/** @type {import('eslint').Linter.Config} */
export default [
  js.configs.recommended,
  {
    languageOptions: { ecmaVersion: 2022, sourceType: "module" },
    rules: {
      "no-console": "warn",
      "no-var": "error",
      "prefer-const": "error",
    },
  },
];
