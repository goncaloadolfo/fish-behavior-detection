import logo from "../images/oceanarium-logo.png";
import "../css/login.css";
import { useState } from "react";
import { useHistory } from "react-router";
import $ from "jquery";

const Login = () => {
  const history = useHistory();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleFieldChange = (e) => {
    e.preventDefault();
    if (e.target.id === "username-input") setUsername(e.target.value);
    else if (e.target.id === "pass-input") setPassword(e.target.value);
  };

  const handleLoginSubmission = (e) => {
    e.preventDefault();
    let usernameInput = $("#username-input");
    let passwordInput = $("#pass-input");

    let invalidUsername = $("#invalid-username");
    let invalidPassword = $("#invalid-password");
    let invalidCredentials = $("#invalid-login");

    if (username.trim() === "") {
      usernameInput.css("border", "1.5px solid red");
      invalidUsername.css("display", "block");
    } else {
      usernameInput.css("border", "1.5px solid gray");
      invalidUsername.css("display", "none");
    }

    if (password === "") {
      passwordInput.css("border", "1.5px solid red");
      invalidPassword.css("display", "block");
    } else {
      passwordInput.css("border", "1.5px solid gray");
      invalidPassword.css("display", "none");
    }

    if (username.trim() !== "" && password !== "") {
      if (username === "abc" && password === "abc")
        history.push("/videos-list");
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
        <form>
          <p id="invalid-login" className="invalid-feedback">
            Invalid credentials.
          </p>
          <input
            id="username-input"
            type="username"
            placeholder="username"
            onChange={handleFieldChange}
          ></input>
          <br />
          <p id="invalid-username" className="invalid-feedback">
            Provide a username.
          </p>
          <input
            id="pass-input"
            type="password"
            placeholder="password"
            onChange={handleFieldChange}
          ></input>
          <br />
          <p id="invalid-password" className="invalid-feedback">
            Provide a password.
          </p>
          <button
            id="login-btn"
            onClick={handleLoginSubmission}
            className="btn btn-outline-secondary"
          >
            Login
          </button>
        </form>
        <div className="text-center">
          <span id="newAccountSpan">Ask for an account</span>
          <span id="forgotMyPassSpan">Forgot my password</span>
        </div>
      </div>
    </div>
  );
};

export default Login;
