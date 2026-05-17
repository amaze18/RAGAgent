# AWS EC2 Guidelines: Hosting & Running Code from a Git Repository

This document is an augmented, **practical guideline** for using **AWS EC2** to host and run code pulled from a **Git repository**. It builds on your existing notes and adds missing conceptual clarity, best practices, and real-world DevOps discipline.

---

## 1. Purpose of This Document

By the end of this guide, you should clearly understand:

- What EC2 is actually responsible for (and what it is not)
- How SSH authentication works across **different trust boundaries**
- How to correctly clone and run code from GitHub on EC2
- Common beginner mistakes (explicitly called out)
- A clean, repeatable mental model you can reuse for any EC2-based project

This is not a click-through tutorial. It is an **operational guideline**.

---

## 2. AWS Account Creation (One-Time Setup)

- Create an AWS account at https://aws.amazon.com/
- Requires:
  * Email & phone number
  * Payment method
  * Government ID (PAN / Voter ID, region-dependent)

⚠️ **Guideline**: Treat your AWS account like production infra. Enable MFA early.

---

## 3. Launching an EC2 Instance (Infrastructure Layer)

### 3.1 What EC2 Actually Is

EC2 = **Virtual machine you fully control**

You are responsible for:

- OS updates
- Security
- Runtime
- Application lifecycle

AWS is only responsible for:

- Hardware
- Hypervisor
- Networking primitives

---

### 3.2 Instance Creation (Ubuntu VM)

From AWS Console:

1. Open **EC2 → Instances → Launch instance**
2. Choose:
   - AMI: Ubuntu LTS
   - Instance type: t2.micro (free tier)
3. Key pair:
   - Create or use existing `.pem` key (example: `zeus.pem`)
4. Network settings:
   - Allow SSH (port 22)
   - Source: Your IP (not 0.0.0.0/0 in real projects)

You correctly noticed similarities with Linux server concepts — this is expected.

---

## 4. SSH Access to EC2 (Machine Authentication)

### 4.1 What `zeus.pem` Is Used For

`zeus.pem` is **ONLY** for:

➡️ Authenticating **you → EC2**

It has **nothing** to do with GitHub.

This distinction is critical.

---

### 4.2 Linux Permissions (on Linux/macOS)

```
chmod 400 zeus.pem
```

---

### 4.3 Windows Permissions (PowerShell – Correct Way)

Windows uses ACLs, not chmod.

```
icacls zeus.pem /inheritance:r
icacls zeus.pem /remove "BUILTIN\Users"
icacls zeus.pem /remove "NT AUTHORITY\Authenticated Users"
icacls zeus.pem /grant:r "$($env:USERNAME):(R)"
```

Verify:

```
icacls zeus.pem
```

Expected: only **your user** with `(R)`

---

### 4.4 SSH into EC2

```
ssh -i zeus.pem ubuntu@<public-dns>
```

Example:

```
ssh -i zeus.pem ubuntu@ec2-16-16-210-255.eu-north-1.compute.amazonaws.com
```

---

## 5. GitHub Access from EC2 (Service Authentication)

### 5.1 Common Beginner Mistake (Explicit)

❌ Trying to use `zeus.pem` for GitHub

This will **never work**.

Why?

- EC2 key = AWS trust domain
- GitHub = separate trust domain

Same protocol (SSH), different identities.

---

### 5.2 Correct Mental Model

| Purpose | Key Used |
| --- | --- |
| Login to EC2 | zeus.pem |
| Access GitHub | ~/.ssh/id_ed25519 |

---

## 6. Setting Up Git on EC2

Run **inside EC2**:

```
git --version
```

If missing:

```
sudo apt update && sudo apt install git -y
```

---

## 7. Generate SSH Key on EC2 (for GitHub)

```
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press **Enter** for all prompts.

Creates:

- `~/.ssh/id_ed25519`
- `~/.ssh/id_ed25519.pub`

---

### 7.1 Add Key to SSH Agent

```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

---

### 7.2 Add Public Key to GitHub

```
cat ~/.ssh/id_ed25519.pub
```

Copy output → GitHub:

Settings → SSH and GPG Keys → New SSH Key

- Title: EC2 Ubuntu
- Key: pasted content

---

### 7.3 Verify Connection (Do NOT Skip)

```
ssh -T git@github.com
```

Expected:

```
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

---

## 8. Clone Repository

```
git clone git@github.com:amaze18/freeGPT.git
```

Then:

```
cd freeGPT
```

---

## 9. Running Code on EC2 (Application Layer)

At this point:

- Infra ✅
- Access ✅
- Source code ✅

Next steps depend on the project:

- Identify runtime (Python / Node / Docker)
- Install dependencies
- Run as:
  * foreground (testing)
  * background (systemd / tmux)

⚠️ **Guideline**: Never jump to Docker or Nginx without understanding why.

---

## 10. Operational Discipline (Non-Negotiable)

- Keep infra keys and app keys separate
- Never expose private keys in repos
- Understand *why* something works, not just *that* it works

You are doing real infrastructure work. Treat it as such.

---

## 11. Next Logical Extensions

When ready, extend this document with:

- Static IP (Elastic IP)
- systemd service files
- Reverse proxy (Nginx)
- Environment variables & secrets
- CI/CD automation

---

## 12. Environment Variables & Secret Management

Once your code is running on EC2, one of the first real-world concerns is keeping secrets — API keys, database credentials, tokens — out of your repository. The correct pattern is to write them directly into a `.env` file on the instance itself, never committing them to Git.

```bash
echo "GOOGLE_API_KEY=sk-your-actual-key-here" > .env
```

This creates (or overwrites) a `.env` file in the current directory with your key. Your application then loads this at runtime via a library like `python-dotenv`. The `.env` file should always be listed in your `.gitignore` so it is never accidentally pushed upstream.

⚠️ **Guideline**: Treat every API key like a password. If it ends up in a public repo — even briefly — rotate it immediately.

---

## 13. Installing Python 3 and pip on Amazon Linux 2

If your project is Python-based and you're running an Amazon Linux 2 AMI (rather than Ubuntu), the package manager is `yum`, not `apt`. The setup sequence is:

```bash
# Update the package list first
sudo yum update -y

# Install Python 3 (pip3 is bundled)
sudo yum install python3 -y

# Confirm pip is available
pip3 --version
```

**If `yum` doesn't give you the version you need**, fall back to the official bootstrap script:

```bash
curl -O https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```

**If pip is installed but the shell says "command not found"**, the binary exists but isn't on your `PATH`. Find it and add it:

```bash
which pip3          # or: find /usr -name pip3
echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
source ~/.bashrc
```

---

## 14. Understanding Linux File Permissions (chmod 400 Explained)

Linux uses a three-digit permission system. Each digit represents a class of user — Owner, Group, and Others — and is the sum of three rights:

| Value | Right |
|-------|-------|
| 4 | Read (r) |
| 2 | Write (w) |
| 1 | Execute (x) |
| 0 | No permission |

So `chmod 400` breaks down as:

| Digit | Target | Calculation | Result |
|-------|--------|-------------|--------|
| First digit (4) | Owner | 4 + 0 + 0 | Read-only |
| Second digit (0) | Group | 0 + 0 + 0 | No access |
| Third digit (0) | Others | 0 + 0 + 0 | No access |

This is exactly what you want for a `.pem` key file: only you can read it, nobody else can touch it. SSH will actually refuse to use the key if the permissions are too permissive — this is a security feature, not a bug.

---

## 15. Exposing a Streamlit App via EC2 Security Groups

By default, EC2 blocks all inbound traffic except what you explicitly allow. Streamlit runs on port **8501**, so you must open that port in your instance's Security Group before the app is reachable from a browser.

**Step-by-step:**

1. In the AWS Console, go to **EC2 → Instances** and select your running instance.
2. Click the **Security** tab in the details panel, then click the linked Security Group ID.
3. Go to **Inbound rules → Edit inbound rules → Add rule**.
4. Configure the new rule:
   - **Type**: Custom TCP
   - **Port range**: 8501
   - **Source**: `0.0.0.0/0` (public access) or **My IP** (recommended for development)
5. Click **Save rules**.

Once saved, your Streamlit app is accessible at:

```
http://<your-ec2-public-ip>:8501
```

⚠️ **Guideline**: Use **My IP** as the source during development. Switch to a specific CIDR block or a load balancer when moving to production — avoid leaving `0.0.0.0/0` open permanently on application ports.

---

**End of Guidelines**
