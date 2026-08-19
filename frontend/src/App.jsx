import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [demoUsers, setDemoUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState("");
  const [history, setHistory] = useState([]);

  const [recommendations, setRecommendations] = useState([]);
  const [briefing, setBriefing] = useState("");

  const [loadingRecommendations, setLoadingRecommendations] = useState(false);

  const [loadingBriefing, setLoadingBriefing] = useState(false);

  const [error, setError] = useState("");

  // ==============================
  // Load demo users
  // ==============================

  const loadDemoUsers = async () => {
    try {
      const response = await fetch(`${API_URL}/demo-users`);

      if (!response.ok) {
        throw new Error("Could not load demo users.");
      }

      const data = await response.json();

      setDemoUsers(data);

      if (data.length > 0) {
        const firstUser = data[0].user_id;

        setSelectedUser(firstUser);

        await loadDemoUser(firstUser);
      }
    } catch (err) {
      console.error("Demo users error:", err);

      setError("Failed to load demo users.");
    }
  };

  // ==============================
  // Load selected user history
  // ==============================

  const loadDemoUser = async (userId) => {
    setError("");

    // clear old output whenever user changes
    setRecommendations([]);
    setBriefing("");

    if (userId === "new-user") {
      setHistory([]);
      return;
    }

    try {
      const response = await fetch(`${API_URL}/demo-user/${userId}`);

      if (!response.ok) {
        throw new Error("Could not load user.");
      }

      const data = await response.json();

      setHistory(data.history || []);
    } catch (err) {
      console.error("Demo user error:", err);

      setHistory([]);

      setError("Failed to load selected user.");
    }
  };

  // ==============================
  // Initial load
  // ==============================

  useEffect(() => {
    loadDemoUsers();
  }, []);

  // ==============================
  // Change selected user
  // ==============================

  const handleUserChange = async (event) => {
    const userId = event.target.value;

    setSelectedUser(userId);

    await loadDemoUser(userId);
  };

  // ==============================
  // Get recommendations
  // ==============================

  const getRecommendations = async () => {
    setLoadingRecommendations(true);

    setError("");

    // clear old briefing because
    // recommendations may change
    setBriefing("");

    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          history: history,
          top_k: 10,
        }),
      });

      if (!response.ok) {
        const text = await response.text();

        throw new Error(text);
      }

      const data = await response.json();

      console.log("RECOMMENDATION RESPONSE:", data);

      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error("Recommendation error:", err);

      setError("Failed to generate recommendations.");
    } finally {
      setLoadingRecommendations(false);
    }
  };

  // ==============================
  // Generate AI briefing
  // ==============================

  const getBriefing = async () => {
    setLoadingBriefing(true);
    setError("");

    // VERY IMPORTANT:
    // remove stale previous briefing
    setBriefing("");

    try {
      console.log("Sending history:", history);

      const response = await fetch(`${API_URL}/briefing`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          history: history,
          top_k: 5,
        }),
      });

      if (!response.ok) {
        const text = await response.text();

        throw new Error(text);
      }

      const data = await response.json();

      console.log("FULL BRIEFING RESPONSE:", data);

      console.log("BRIEFING TEXT:", data.briefing);

      const newBriefing = data.briefing || "";

      setBriefing(newBriefing);

      // Give React time to render
      // the new briefing section
      setTimeout(() => {
        const briefingElement = document.querySelector(".briefing");

        if (briefingElement) {
          briefingElement.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }
      }, 250);
    } catch (err) {
      console.error("Briefing error:", err);

      setError("Failed to generate AI briefing.");
    } finally {
      setLoadingBriefing(false);
    }
  };

  // ==============================
  // Cold-start check
  // ==============================

  const isColdStart = selectedUser === "new-user";

  return (
    <main className="app">
      {/* =========================
          HERO
      ========================= */}

      <section className="hero">
        <h1>EngageRank</h1>

        <p>
          Personalized news discovery powered by semantic retrieval, adaptive
          ranking, diversity reranking and grounded AI briefings.
        </p>
      </section>

      {/* =========================
          USER PANEL
      ========================= */}

      <section className="input-panel">
        <label className="input-label">Demo User</label>

        <select
          className="user-select"
          value={selectedUser}
          onChange={handleUserChange}
        >
          {demoUsers.map((user) => (
            <option key={user.user_id} value={user.user_id}>
              {user.user_id}
              {" — "}
              {user.history_size}
              {" clicks"}
            </option>
          ))}

          <option value="new-user">New User — Cold Start</option>
        </select>

        {/* USER INFORMATION */}

        <div className="user-info">
          {isColdStart ? (
            <>
              <strong>Cold-start user</strong>

              <span>
                No previous reading history. Popularity-based recommendations
                will be used.
              </span>
            </>
          ) : (
            <>
              <strong>
                {history.length}
                {" previous clicks"}
              </strong>

              <span>
                EngageRank will use this reading history to build the user's
                interest profile.
              </span>
            </>
          )}
        </div>

        {/* BUTTONS */}

        <div className="actions">
          <button
            className="primary-btn"
            onClick={getRecommendations}
            disabled={loadingRecommendations || loadingBriefing}
          >
            {loadingRecommendations
              ? "Ranking Articles..."
              : "Get Recommendations"}
          </button>

          <button
            className="secondary-btn"
            onClick={getBriefing}
            disabled={loadingBriefing || loadingRecommendations}
          >
            {loadingBriefing
              ? "Generating Briefing..."
              : isColdStart
                ? "Generate News Briefing"
                : "Generate AI Briefing"}
          </button>
        </div>
      </section>

      {/* =========================
          ERROR MESSAGE
      ========================= */}

      {error && <div className="error-message">{error}</div>}

      {/* =========================
          RECOMMENDATIONS
      ========================= */}

      {recommendations.length > 0 && (
        <section>
          <div className="section-header">
            <h2>
              {isColdStart ? "Trending recommendations" : "Recommended for you"}
            </h2>

            <span>
              {recommendations.length}
              {" articles"}
            </span>
          </div>

          <div className="grid">
            {recommendations.map((item, index) => (
              <article className="card" key={item.news_id}>
                <span className="rank">#{index + 1}</span>

                <div className="meta">
                  <span className="badge">{item.category}</span>

                  <span className="badge">
                    {item.mode === "cold_start" ? "Cold Start" : "Personalized"}
                  </span>
                </div>

                <h3>{item.title}</h3>

                <div className="score">
                  {item.mode === "personalized"
                    ? "Relevance score"
                    : "Popularity score"}
                  :{" "}
                  {item.mode === "personalized"
                    ? item.score.toFixed(4)
                    : Math.round(item.score)}
                </div>
              </article>
            ))}
          </div>
        </section>
      )}

      {/* =========================
          AI BRIEFING
      ========================= */}

      {briefing && (
        <section className="briefing">
          <div className="section-header">
            <h2>
              {isColdStart ? "AI News Briefing" : "AI Personalized Briefing"}
            </h2>

            <span>Grounded in recommended articles</span>
          </div>

          <div className="briefing-content">{briefing}</div>
        </section>
      )}
    </main>
  );
}

export default App;
