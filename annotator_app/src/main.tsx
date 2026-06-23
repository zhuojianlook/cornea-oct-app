import React from "react";
import ReactDOM from "react-dom/client";
import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";
import App from "./App";
import "./styles/globals.css";

// Dark MUI theme — WITHOUT this, MUI components (menus, fields, toggle buttons, Typography) fall
// back to their default LIGHT palette and render near-black text on our dark surfaces (unreadable).
// Mirrors the main cornea_app theme so the two apps look consistent. text.secondary is brightened
// (#a6a6ac) for legible secondary labels on dark.
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#7aa6d6" },
    secondary: { main: "#63a66a" },
    error: { main: "#ff453a" },
    background: { default: "#1c1c1e", paper: "#2c2c2e" },
    text: { primary: "#e5e5ea", secondary: "#a6a6ac" },
  },
  typography: {
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    fontSize: 13,
  },
  components: {
    MuiButton: {
      defaultProps: { size: "small", disableElevation: true },
      styleOverrides: { root: { textTransform: "none", fontSize: "0.75rem" } },
    },
    MuiSelect: { defaultProps: { size: "small" } },
    MuiSlider: { defaultProps: { size: "small" } },
    MuiTextField: { defaultProps: { size: "small", variant: "outlined" } },
    MuiDialog: { styleOverrides: { paper: { backgroundImage: "none" } } },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>,
);
