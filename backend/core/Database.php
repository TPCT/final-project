<?php
namespace core;

use PDO;

class Database{
    private ?PDO $connection;

    public function __construct($config){
        $db_connection = $config['DB_CONNECTION'] ?? '';
        $db_host = $config['DB_HOST'] ?? '';
        $db_name = $config['DB_NAME'] ?? '';
        $db_user = $config['DB_USER'] ?? '';
        $db_pass = $config['DB_PASS'] ?? '';

        if (!$db_host || !$db_name || !$db_connection){
            Application::APP()->error_logger->log(0, "couldn't connect to the database using configuration file: \n\t\tparameters = {connection: $db_connection, host: $db_host, db_name: $db_name}", __FILE__, 17);
            $this->connection = Null;
            return $this;
        }

        $this::CREATE_DB($config);

        $dsn = "{$db_connection}:host={$db_host};dbname={$db_name}";
        $this->connection = new PDO($dsn, $db_user, $db_pass,[
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_EMULATE_PREPARES => false,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        ]);
    }

    public function connector(): ?PDO{
        return $this->connection;
    }

    public static function CREATE_DB(array $config){
        if ($config){
            $db_connection = $config['DB_CONNECTION'] ?? '';
            $db_host = $config['DB_HOST'] ?? '';
            $db_name = $config['DB_NAME'] ?? '';
            $db_user = $config['DB_USER'] ?? '';
            $db_pass = $config['DB_PASS'] ?? '';
            if (!$db_connection || $db_host)
                return Null;

            $dsn = "{$db_connection}:host={$db_host};";

            $connector = new PDO($dsn, $db_user, $db_pass,[
                PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION
            ]);

            $query = "CREATE DATABASE IF NOT EXISTS `:db`;";
            $stmt = $connector->prepare($query);
            $stmt->bindParam(":db", $db_name);
            return $stmt->execute();
        }
    }

    public function applyMigrations(){
        $appliedMigrations = array();
        $this->createMigrationsTable();
        $applied_migrations = $this->getAppliedMigrations();

        $files = \scandir(Application::ROOT_DIR() . \DIRECTORY_SEPARATOR . "migrations");
        $toApplyMigrations = \array_diff($files, $applied_migrations);
        foreach($toApplyMigrations as $migration){
            if ($migration === '.' || $migration === '..')
                continue;
            require_once(Application::ROOT_DIR() . \DIRECTORY_SEPARATOR . "migrations" . \DIRECTORY_SEPARATOR . $migration);
            $class = "migrations\\" . \pathinfo($migration, \PATHINFO_FILENAME);
            $class = new $class();
            if ($class->up())
                $appliedMigrations[] = $migration;
        }

        if ($appliedMigrations)
            $this->saveMigrations($appliedMigrations);

    }

    public function getAppliedMigrations(): bool|array{
        $query = "SELECT migration FROM migrations";
        $stmt = $this->connection->prepare($query);
        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_COLUMN);
    }

    public function createMigrationsTable(){
        if (!$this->connection)
            return;

        $this->connection->exec(
            "CREATE TABLE IF NOT EXISTS migrations(
            id INT AUTO_INCREMENT PRIMARY KEY,
            migration VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP) ENGINE=INNODB;"
        );
    }

    public function saveMigrations(array $migrations){
        foreach($migrations as $migration){
            $query = "INSERT INTO migrations(migration) VALUES (:migration_name)";
            $stmt = $this->connection->prepare($query);
            $stmt->bindParam(":migration_name", $migration);
            $stmt->execute();
        }
    }
}