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
  - Email & phone number
  - Payment method
  - Government ID (PAN / Voter ID, region-dependent)

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

```bash
chmod 400 zeus.pem
```

---

### 4.3 Windows Permissions (PowerShell – Correct Way)

Windows uses ACLs, not chmod.

```powershell
icacls zeus.pem /inheritance:r
icacls zeus.pem /remove "BUILTIN\Users"
icacls zeus.pem /remove "NT AUTHORITY\Authenticated Users"
icacls zeus.pem /grant:r "$($env:USERNAME):(R)"
```

Verify:
```powershell
icacls zeus.pem
```

Expected: only **your user** with `(R)`

---

### 4.4 SSH into EC2

```bash
ssh -i zeus.pem ubuntu@<public-dns>
```

Example:
```bash
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
|------|--------|
| Login to EC2 | zeus.pem |
| Access GitHub | ~/.ssh/id_ed25519 |

---

## 6. Setting Up Git on EC2

Run **inside EC2**:

```bash
git --version
```

If missing:
```bash
sudo apt update && sudo apt install git -y
```

---

## 7. Generate SSH Key on EC2 (for GitHub)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press **Enter** for all prompts.

Creates:
- `~/.ssh/id_ed25519`
- `~/.ssh/id_ed25519.pub`

---

### 7.1 Add Key to SSH Agent

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

---

### 7.2 Add Public Key to GitHub

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy output → GitHub:

Settings → SSH and GPG Keys → New SSH Key

- Title: EC2 Ubuntu
- Key: pasted content

---

### 7.3 Verify Connection (Do NOT Skip)

```bash
ssh -T git@github.com
```

Expected:
```
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

---

## 8. Clone Repository

```bash
git clone git@github.com:amaze18/freeGPT.git
```

Then:
```bash
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
  - foreground (testing)
  - background (systemd / tmux)

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

**End of Guidelines**

