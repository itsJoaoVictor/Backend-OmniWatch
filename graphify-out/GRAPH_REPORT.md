# Graph Report - Backend-OmniWatch  (2026-08-21)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 39 nodes · 50 edges · 10 communities (8 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- router.py
- env.py
- services.py
- database.py
- backend-omniwatch

## God Nodes (most connected - your core abstractions)
1. `UserCreate` - 7 edges
2. `create_user()` - 7 edges
3. `run_async_migrations()` - 4 edges
4. `User` - 4 edges
5. `register_user()` - 4 edges
6. `do_run_migrations()` - 3 edges
7. `run_migrations_online()` - 3 edges
8. `get_password_hash()` - 3 edges
9. `UserResponse` - 3 edges
10. `run_migrations_offline()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `register_user()` --calls--> `create_user()`  [EXTRACTED]
  app/users/router.py → app/users/services.py
- `create_user()` --references--> `UserCreate`  [EXTRACTED]
  app/users/services.py → app/users/schemas.py
- `create_user()` --calls--> `get_password_hash()`  [EXTRACTED]
  app/users/services.py → app/core/security.py
- `create_user()` --references--> `User`  [EXTRACTED]
  app/users/services.py → app/users/models.py
- `register_user()` --references--> `UserCreate`  [EXTRACTED]
  app/users/router.py → app/users/schemas.py

## Import Cycles
- None detected.

## Communities (10 total, 2 thin omitted)

### Community 0 - "router.py"
Cohesion: 0.29
Nodes (5): AsyncSession, register_user(), UserCreate, UserResponse, BaseModel

### Community 1 - "env.py"
Cohesion: 0.28
Nodes (8): do_run_migrations(), Run migrations in 'online' mode., Run migrations in 'offline' mode.      This configures the context with just a U, In this scenario we need to create an Engine     and associate a connection with, run_async_migrations(), run_migrations_offline(), run_migrations_online(), Connection

### Community 2 - "services.py"
Cohesion: 0.33
Nodes (5): get_password_hash(), User, create_user(), AsyncSession, Base

## Knowledge Gaps
- **1 isolated node(s):** `backend-omniwatch`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UserCreate` connect `router.py` to `services.py`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `create_user()` connect `services.py` to `router.py`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **What connects `backend-omniwatch` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._