import logo from "../images/oceanarium-logo.png";
import "../css/login.css";
import { Link } from "react-router-dom";
import { useState } from "react";
import $ from "jquery";

const Login = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleFieldChange = (e) => {
    e.preventDefault();
    if (e.target.type === "username") setUsername(e.target.value);
    else setPassword(e.target.value);
  };

  const handleLoginSubmission = () => {
    if (username.trim === "") {
      // not working...
      $("#username-input").style.border = "2px red";
    }
    if (password === "") {
      // not working...
      $("#password-input").style.border = "2px red";
    }
  };

  return (
    <div
      className="d-flex justify-content-center align-items-center"
      style={{ height: window.innerHeight }}
    >
      <div id="login-container">
        <img src={logo} width="20%" alt="oceanarium logo"></img>
        <h2 className="fw-bold">Fish Behavior Detection</h2>
        <form onSubmit={handleLoginSubmission}>
          <p id="invalid-login" className="invalid-feedback">
            Invalid credentials.
          </p>
          <input
            id="username-input"
            type="username"
            placeholder="username"
            onChange={handleFieldChange}
          ></input>
          <p id="invalid-username" className="invalid-feedback">
            Provide a username.
          </p>
          <input
            id="pass-input"
            type="password"
            placeholder="password"
            onChange={handleFieldChange}
          ></input>
          <p id="invalid-password" className="invalid-feedback">
            Provide a password.
          </p>
          <button
            id="login-btn"
            type="submit"
            className="btn btn-outline-secondary"
          >
            Login
          </button>
        </form>
        <div className="text-center">
          <Link id="newAccountLink" to="/">
            Ask for an account
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Login;
